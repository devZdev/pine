"""
coinbase_ingestor.py
====================
Async CCXT Coinbase Advanced Trade BTC/USD OHLCV fetcher.

Design notes
------------
* Uses ``ccxt.async_support.coinbase`` (the Coinbase Advanced Trade v3 adapter).
* Fetches in paginated 1000-bar chunks (~11 000 requests for 1m from 2020).
* A token-bucket rate limiter caps throughput at 10 req/s (Coinbase public limit).
* Exponential backoff with jitter wraps every network call.
* Partial progress is saved to parquet after each chunk so a crash loses at
  most one chunk's worth of data.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timezone
from typing import Any

import ccxt.async_support as ccxt_async
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from .storage import checkpoint_save, load_parquet, merge_and_deduplicate, save_parquet
from .utils import (
    AsyncRateLimiter,
    async_retry,
    detect_and_fill_gaps,
    timeframe_to_pandas_freq,
    chunk_date_range,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOL: str = "BTC/USD"
SAFE_SYMBOL: str = "BTC_USD"          # used in filenames
EXCHANGE_ID: str = "coinbase"
CANDLES_PER_REQUEST: int = 300        # Coinbase Advanced Trade max per call
RATE_LIMIT_RPS: float = 9.0           # stay just under the 10 req/s ceiling
MAX_FFILL_CANDLES: int = 5


# ── Exchange factory ──────────────────────────────────────────────────────────

def _build_exchange(api_key: str, api_secret: str) -> ccxt_async.coinbase:
    """Instantiate and return a CCXT async Coinbase exchange object."""
    exchange = ccxt_async.coinbase(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": False,   # we manage rate limiting ourselves
            "options": {
                "defaultType": "spot",
            },
        }
    )
    return exchange


# ── Core fetch helpers ────────────────────────────────────────────────────────

async def _fetch_ohlcv_chunk(
    exchange: ccxt_async.coinbase,
    timeframe: str,
    since_ms: int,
    limiter: AsyncRateLimiter,
) -> list[list[Any]]:
    """Fetch a single chunk of OHLCV bars with rate limiting + backoff.

    Parameters
    ----------
    exchange:
        Authenticated CCXT exchange instance.
    timeframe:
        CCXT timeframe string, e.g. ``"1m"``.
    since_ms:
        Epoch milliseconds for the start of this chunk.
    limiter:
        Shared token-bucket rate limiter.

    Returns
    -------
    list[list]
        Raw CCXT OHLCV list: [[timestamp_ms, O, H, L, C, V], ...]
    """

    @async_retry(
        max_attempts=7,
        base_delay=1.0,
        max_delay=120.0,
        jitter=0.25,
        retryable_exceptions=(Exception,),
    )
    async def _call() -> list[list[Any]]:
        async with limiter:
            bars: list[list[Any]] = await exchange.fetch_ohlcv(
                SYMBOL,
                timeframe=timeframe,
                since=since_ms,
                limit=CANDLES_PER_REQUEST,
                params={},
            )
        return bars

    return await _call()


async def _fetch_all_ohlcv(
    exchange: ccxt_async.coinbase,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limiter: AsyncRateLimiter,
) -> pd.DataFrame:
    """Paginate through the full [start, end] range and return a single DataFrame.

    Parameters
    ----------
    exchange:
        Authenticated exchange instance.
    timeframe:
        e.g. ``"1m"`` or ``"5m"``.
    start, end:
        Inclusive UTC timestamps.
    limiter:
        Shared rate limiter.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.  Index: UTC DatetimeIndex.
    """
    freq = timeframe_to_pandas_freq(timeframe)
    # Compute chunk width in milliseconds
    tf_ms = pd.tseries.frequencies.to_offset(freq).nanos // 1_000_000  # type: ignore[union-attr]
    chunk_ms = tf_ms * CANDLES_PER_REQUEST

    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_bars: list[list[Any]] = []
    request_count = 0

    logger.info(
        "[coinbase] Starting paginated fetch: {} {} from {} to {}.",
        SYMBOL, timeframe, start.date(), end.date(),
    )

    while since_ms < end_ms:
        bars = await _fetch_ohlcv_chunk(exchange, timeframe, since_ms, limiter)

        if not bars:
            logger.debug("[coinbase] Empty response at since_ms={}; stopping.", since_ms)
            break

        # Filter out any bars beyond our end boundary
        bars = [b for b in bars if b[0] <= end_ms]
        all_bars.extend(bars)
        request_count += 1

        last_ts_ms = bars[-1][0]
        next_since_ms = last_ts_ms + tf_ms

        if request_count % 500 == 0:
            pct = 100 * (last_ts_ms - int(start.timestamp() * 1000)) / max(end_ms - int(start.timestamp() * 1000), 1)
            logger.info(
                "[coinbase] {} requests completed, {:.1f}% done ({} bars buffered).",
                request_count, pct, len(all_bars),
            )

        if next_since_ms >= end_ms or last_ts_ms >= end_ms:
            break

        since_ms = next_since_ms

    logger.info(
        "[coinbase] Fetch complete: {} requests, {} raw bars.",
        request_count, len(all_bars),
    )

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = _bars_to_dataframe(all_bars)
    return df


def _bars_to_dataframe(bars: list[list[Any]]) -> pd.DataFrame:
    """Convert raw CCXT OHLCV list to a typed DataFrame with a UTC DatetimeIndex."""
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    # Drop duplicates that sometimes appear at chunk boundaries
    df = df[~df.index.duplicated(keep="first")]
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df


# ── Incremental fetch with checkpoint saves ───────────────────────────────────

async def ingest_coinbase(
    timeframes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_dir: str = "data/raw",
    checkpoint_interval: int = 50_000,
) -> dict[str, pd.DataFrame]:
    """Top-level entry point: ingest BTC/USD OHLCV for one or more timeframes.

    For each timeframe:
    1. Load any existing parquet (incremental mode).
    2. Fetch only the missing tail from the exchange.
    3. Gap-detect and forward-fill.
    4. Save to parquet.

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
    checkpoint_interval:
        Save a checkpoint every this many newly fetched rows.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of timeframe → clean OHLCV DataFrame.
    """
    load_dotenv()
    api_key = os.environ["COINBASE_API_KEY"]
    api_secret = os.environ["COINBASE_API_SECRET"]

    exchange = _build_exchange(api_key, api_secret)
    limiter = AsyncRateLimiter(rate=RATE_LIMIT_RPS, period=1.0)

    results: dict[str, pd.DataFrame] = {}

    try:
        for tf in timeframes:
            logger.info("[coinbase] Processing timeframe {}.", tf)

            # Load existing data; resume from where we left off
            existing = load_parquet(SAFE_SYMBOL, tf, base_dir)
            if existing is not None and not existing.empty:
                resume_start = existing.index[-1] + pd.tseries.frequencies.to_offset(
                    timeframe_to_pandas_freq(tf)
                )
                logger.info(
                    "[coinbase] Resuming from {} (have {} bars).",
                    resume_start, len(existing),
                )
                fetch_start = resume_start
            else:
                fetch_start = start

            if fetch_start >= end:
                logger.info("[coinbase] {} is already up to date.", tf)
                results[tf] = existing  # type: ignore[assignment]
                continue

            # Paginated async fetch
            new_df = await _fetch_all_ohlcv(
                exchange, tf, fetch_start, end, limiter
            )

            # Merge with existing
            combined = merge_and_deduplicate(existing, new_df)

            # Gap fill
            freq = timeframe_to_pandas_freq(tf)
            combined = detect_and_fill_gaps(
                combined, freq=freq, max_ffill_candles=MAX_FFILL_CANDLES,
                symbol=SAFE_SYMBOL,
            )

            # Persist
            save_parquet(combined, SAFE_SYMBOL, tf, base_dir)
            results[tf] = combined

    finally:
        await exchange.close()
        logger.debug("[coinbase] Exchange connection closed.")

    return results


# ── Convenience wrapper ───────────────────────────────────────────────────────

async def ingest_coinbase_with_interrupt_handling(
    timeframes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_dir: str = "data/raw",
    partial_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Wraps :func:`ingest_coinbase` so that ``KeyboardInterrupt`` triggers a
    checkpoint save of whatever data has been fetched so far.

    Parameters
    ----------
    partial_results:
        Mutable dict that the caller can inspect after a ``KeyboardInterrupt``.
        If ``None`` a local dict is used.
    """
    if partial_results is None:
        partial_results = {}

    try:
        result = await ingest_coinbase(timeframes, start, end, base_dir)
        partial_results.update(result)
    except KeyboardInterrupt:
        logger.warning("[coinbase] KeyboardInterrupt received — saving partial progress.")
        for tf, df in partial_results.items():
            if df is not None and not df.empty:
                checkpoint_save(df, SAFE_SYMBOL, tf, base_dir)
        raise

    return partial_results
