"""
alpaca_ingestor.py
==================
Async Alpaca Markets TSLA OHLCV fetcher (paper trading API).

Design notes
------------
* Uses ``alpaca-py`` (``alpaca.data.historical.StockHistoricalDataClient``) for
  authenticated requests, but drives pagination manually via ``aiohttp`` to
  keep the fully-async architecture consistent with the Coinbase ingestor.
* Free-tier rate limit: 200 req/min → ~3.33 req/s.  We target 3 req/s with
  the token-bucket limiter plus exponential backoff.
* Market hours only: 09:30–16:00 Eastern Time on NYSE trading days.
  We filter out non-trading timestamps *after* fetching — Alpaca's API
  already returns only market-hours bars, but we validate defensively.
* Pagination uses Alpaca's ``next_page_token`` cursor (v2 Bars API).
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timezone, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from .storage import checkpoint_save, load_parquet, merge_and_deduplicate, save_parquet
from .utils import (
    AsyncRateLimiter,
    async_retry,
    detect_and_fill_gaps,
    timeframe_to_pandas_freq,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOL: str = "TSLA"
ALPACA_BASE_URL: str = "https://data.alpaca.markets/v2"
BARS_ENDPOINT: str = f"{ALPACA_BASE_URL}/stocks/{{symbol}}/bars"

# Alpaca timeframe notation
_TF_MAP: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "1h": "1Hour",
    "1d": "1Day",
}

LIMIT_PER_PAGE: int = 10_000          # Alpaca v2 max per request
RATE_LIMIT_RPS: float = 3.0           # free tier ≈ 200/min → 3.33/s; use 3
MAX_FFILL_CANDLES: int = 5

ET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN_ET = (9, 30)   # 09:30 ET
MARKET_CLOSE_ET = (16, 0)  # 16:00 ET


# ── NYSE trading calendar ─────────────────────────────────────────────────────

def _is_trading_minute(ts: pd.Timestamp) -> bool:
    """Return True if *ts* falls within NYSE market hours (09:30–15:59 ET)
    on a weekday.  Holidays are NOT checked (Alpaca omits them automatically).
    """
    ts_et = ts.tz_convert(ET_TZ)
    if ts_et.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    market_open = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = ts_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= ts_et < market_close


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows outside NYSE market hours."""
    if df.empty:
        return df
    mask = pd.Series(df.index, index=df.index).apply(_is_trading_minute)
    dropped = (~mask).sum()
    if dropped:
        logger.debug("[alpaca] Dropped {} non-market-hours bars.", dropped)
    return df.loc[mask]


# ── Async HTTP helpers ────────────────────────────────────────────────────────

def _build_headers(api_key: str, secret_key: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }


async def _fetch_page(
    session: aiohttp.ClientSession,
    symbol: str,
    alpaca_timeframe: str,
    start_rfc3339: str,
    end_rfc3339: str,
    page_token: str | None,
    limiter: AsyncRateLimiter,
) -> dict[str, Any]:
    """Fetch a single page of bars from Alpaca v2 REST API.

    Parameters
    ----------
    session:
        Shared ``aiohttp.ClientSession``.
    symbol:
        Equity ticker, e.g. ``"TSLA"``.
    alpaca_timeframe:
        Alpaca-style timeframe string, e.g. ``"1Min"``.
    start_rfc3339, end_rfc3339:
        ISO-8601 date-time strings (UTC) for the query window.
    page_token:
        Cursor returned by the previous page; ``None`` for the first page.
    limiter:
        Shared token-bucket rate limiter.

    Returns
    -------
    dict
        Parsed JSON response: ``{"bars": [...], "next_page_token": ...}``.
    """
    url = BARS_ENDPOINT.format(symbol=symbol)
    params: dict[str, Any] = {
        "timeframe": alpaca_timeframe,
        "start": start_rfc3339,
        "end": end_rfc3339,
        "limit": LIMIT_PER_PAGE,
        "adjustment": "split",       # corporate action adjustment
        "feed": "iex",               # IEX feed available on free tier
    }
    if page_token:
        params["page_token"] = page_token

    @async_retry(
        max_attempts=7,
        base_delay=1.0,
        max_delay=120.0,
        jitter=0.25,
        retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError, Exception),
    )
    async def _call() -> dict[str, Any]:
        async with limiter:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    logger.warning("[alpaca] 429 Too Many Requests — sleeping {:.1f}s.", retry_after)
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=429,
                        message="Rate limited",
                    )
                resp.raise_for_status()
                return await resp.json()  # type: ignore[no-any-return]

    return await _call()


async def _fetch_all_bars(
    session: aiohttp.ClientSession,
    symbol: str,
    alpaca_timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limiter: AsyncRateLimiter,
) -> pd.DataFrame:
    """Paginate through the full [start, end] range using Alpaca's cursor.

    Parameters
    ----------
    session:
        Shared ``aiohttp.ClientSession``.
    symbol:
        Ticker.
    alpaca_timeframe:
        Alpaca-style timeframe, e.g. ``"1Min"``.
    start, end:
        UTC timestamps (inclusive).
    limiter:
        Rate limiter.

    Returns
    -------
    pd.DataFrame
        OHLCV bars indexed by UTC timestamp.
    """
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    all_bars: list[dict[str, Any]] = []
    page_token: str | None = None
    page_num = 0

    logger.info(
        "[alpaca] Starting paginated fetch: {} {} from {} to {}.",
        symbol, alpaca_timeframe, start.date(), end.date(),
    )

    while True:
        payload = await _fetch_page(
            session, symbol, alpaca_timeframe,
            start_str, end_str, page_token, limiter,
        )

        bars_page: list[dict[str, Any]] = payload.get("bars", []) or []
        all_bars.extend(bars_page)
        page_num += 1

        if page_num % 100 == 0:
            logger.info(
                "[alpaca] {} pages fetched, {} bars buffered.",
                page_num, len(all_bars),
            )

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    logger.info(
        "[alpaca] Fetch complete: {} pages, {} raw bars.",
        page_num, len(all_bars),
    )

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    return _bars_to_dataframe(all_bars)


def _bars_to_dataframe(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert Alpaca v2 bar dicts to a typed DataFrame with UTC DatetimeIndex."""
    df = pd.DataFrame(bars)
    # Alpaca returns 't', 'o', 'h', 'l', 'c', 'v', 'n', 'vw'
    rename = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df.rename(columns=rename, inplace=True)

    # Keep only OHLCV
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


# ── Gap fill for market hours ─────────────────────────────────────────────────

def _fill_gaps_market_hours(
    df: pd.DataFrame,
    freq: str,
    symbol: str = "TSLA",
) -> pd.DataFrame:
    """Gap fill that is aware of market hours.

    We cannot simply use a full calendar frequency because that would create
    bars at 02:00 on a weekend.  Instead we:
    1. Build the expected market-hours index for the date range.
    2. Reindex the DataFrame onto that index.
    3. Forward-fill gaps ≤ MAX_FFILL_CANDLES; warn on larger gaps.
    """
    if df.empty:
        return df

    # Build market-hours timestamp grid
    all_minutes = pd.date_range(
        start=df.index[0].normalize(),
        end=df.index[-1].normalize() + pd.Timedelta(days=1),
        freq=freq,
        tz="UTC",
    )
    market_minutes = pd.DatetimeIndex([ts for ts in all_minutes if _is_trading_minute(ts)])

    # Clip to our data range
    market_minutes = market_minutes[
        (market_minutes >= df.index[0]) & (market_minutes <= df.index[-1])
    ]

    missing = market_minutes.difference(df.index)
    if len(missing) > 0:
        logger.debug(
            "[{}] {} missing market-hours candles.",
            symbol, len(missing),
        )
        # Warn about large gaps
        from .utils import _find_gap_runs
        _find_gap_runs(missing, freq=freq, max_ffill_candles=MAX_FFILL_CANDLES, symbol=symbol)

    df = df.reindex(market_minutes)
    df["is_filled"] = df["close"].isna()
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)

    return df


# ── Top-level ingestor ────────────────────────────────────────────────────────

async def ingest_alpaca(
    timeframes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_dir: str = "data/raw",
) -> dict[str, pd.DataFrame]:
    """Top-level entry point: ingest TSLA OHLCV for one or more timeframes.

    Parameters
    ----------
    timeframes:
        e.g. ``["1m", "5m"]``.
    start:
        Historical start (UTC).
    end:
        Historical end (UTC, inclusive).
    base_dir:
        Root directory for parquet files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of timeframe → clean OHLCV DataFrame.
    """
    load_dotenv()
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    headers = _build_headers(api_key, secret_key)
    limiter = AsyncRateLimiter(rate=RATE_LIMIT_RPS, period=1.0)

    results: dict[str, pd.DataFrame] = {}

    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        for tf in timeframes:
            alpaca_tf = _TF_MAP.get(tf)
            if alpaca_tf is None:
                logger.error("[alpaca] Unsupported timeframe '{}'. Skipping.", tf)
                continue

            logger.info("[alpaca] Processing timeframe {}.", tf)

            # Incremental: resume from last saved bar
            existing = load_parquet(SYMBOL, tf, base_dir)
            if existing is not None and not existing.empty:
                freq_offset = pd.tseries.frequencies.to_offset(timeframe_to_pandas_freq(tf))
                resume_start = existing.index[-1] + freq_offset  # type: ignore[operator]
                logger.info(
                    "[alpaca] Resuming from {} (have {} bars).",
                    resume_start, len(existing),
                )
                fetch_start = resume_start
            else:
                fetch_start = start

            if fetch_start >= end:
                logger.info("[alpaca] {} is already up to date.", tf)
                results[tf] = existing  # type: ignore[assignment]
                continue

            new_df = await _fetch_all_bars(
                session, SYMBOL, alpaca_tf, fetch_start, end, limiter
            )

            # Filter to market hours only
            new_df = _filter_market_hours(new_df)

            # Merge with existing
            combined = merge_and_deduplicate(existing, new_df)

            # Gap fill (market-hours aware)
            freq = timeframe_to_pandas_freq(tf)
            combined = _fill_gaps_market_hours(combined, freq=freq, symbol=SYMBOL)

            # Persist
            save_parquet(combined, SYMBOL, tf, base_dir)
            results[tf] = combined

    return results


async def ingest_alpaca_with_interrupt_handling(
    timeframes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_dir: str = "data/raw",
    partial_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Wraps :func:`ingest_alpaca` and saves partial progress on ``KeyboardInterrupt``."""
    if partial_results is None:
        partial_results = {}

    try:
        result = await ingest_alpaca(timeframes, start, end, base_dir)
        partial_results.update(result)
    except KeyboardInterrupt:
        logger.warning("[alpaca] KeyboardInterrupt — saving partial progress.")
        for tf, df in partial_results.items():
            if df is not None and not df.empty:
                checkpoint_save(df, SYMBOL, tf, base_dir)
        raise

    return partial_results
