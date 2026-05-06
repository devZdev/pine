"""
utils.py
========
Shared utilities:
  - Loguru logging setup
  - Token-bucket rate limiter (async)
  - Exponential backoff with jitter decorator
  - OHLCV gap detection and forward-fill repair
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any, Callable, Coroutine, TypeVar

import pandas as pd
from loguru import logger

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str | None = "pipeline.log") -> None:
    """Configure loguru sinks.

    Parameters
    ----------
    level:
        Minimum log level for the console sink.
    log_file:
        Path for the rotating file sink.  Pass ``None`` to disable file logging.
    """
    import sys

    logger.remove()  # drop default handler

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,
    )

    if log_file is not None:
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
            rotation="50 MB",
            retention="14 days",
            compression="gz",
            enqueue=True,
        )

    logger.debug("Logging configured. level={}", level)


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class AsyncRateLimiter:
    """Async token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Maximum number of tokens (requests) per *period* seconds.
    period:
        Window length in seconds (default 1.0 → rate = req/s).
    burst:
        Maximum burst capacity.  Defaults to ``rate``.

    Usage
    -----
    limiter = AsyncRateLimiter(rate=10, period=1.0)  # 10 req/s
    async with limiter:
        await do_request()
    """

    def __init__(self, rate: float, period: float = 1.0, burst: float | None = None) -> None:
        self._rate = rate          # tokens added per period
        self._period = period      # seconds per refill cycle
        self._capacity = burst if burst is not None else rate
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * (self._rate / self._period)
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until *tokens* tokens are available."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Sleep until enough tokens accumulate
                deficit = tokens - self._tokens
                sleep_time = deficit * (self._period / self._rate)
                await asyncio.sleep(sleep_time)

    async def __aenter__(self) -> "AsyncRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass


# ── Exponential backoff with jitter ──────────────────────────────────────────

F = TypeVar("F")


def async_retry(
    max_attempts: int = 7,
    base_delay: float = 1.0,
    max_delay: float = 120.0,
    jitter: float = 0.25,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Coroutine[Any, Any, F]]], Callable[..., Coroutine[Any, Any, F]]]:
    """Decorator: retry an async function with full-jitter exponential backoff.

    The sleep between attempt *k* is::

        min(max_delay, base_delay * 2^k) * U(1 - jitter, 1 + jitter)

    Parameters
    ----------
    max_attempts:
        Total attempts (first call + retries).
    base_delay:
        Starting sleep in seconds.
    max_delay:
        Cap on sleep duration.
    jitter:
        Fraction of jitter added/subtracted from the computed delay.
    retryable_exceptions:
        Only retry on these exception types.
    """

    def decorator(fn: Callable[..., Coroutine[Any, Any, F]]) -> Callable[..., Coroutine[Any, Any, F]]:
        async def wrapper(*args: Any, **kwargs: Any) -> F:
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    if attempt == max_attempts - 1:
                        logger.error(
                            "All {} attempts exhausted for {}: {}",
                            max_attempts, fn.__qualname__, exc,
                        )
                        raise
                    raw_delay = min(max_delay, base_delay * (2 ** attempt))
                    lo = raw_delay * (1 - jitter)
                    hi = raw_delay * (1 + jitter)
                    sleep = random.uniform(lo, hi)
                    logger.warning(
                        "Attempt {}/{} failed for {}: {}. Retrying in {:.2f}s",
                        attempt + 1, max_attempts, fn.__qualname__, exc, sleep,
                    )
                    await asyncio.sleep(sleep)
            # unreachable
            raise RuntimeError("async_retry: unreachable")

        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        return wrapper

    return decorator


# ── Gap detection and forward-fill repair ────────────────────────────────────

OHLCV_COLS: list[str] = ["open", "high", "low", "close", "volume"]


def detect_and_fill_gaps(
    df: pd.DataFrame,
    freq: str,
    max_ffill_candles: int = 5,
    symbol: str = "",
) -> pd.DataFrame:
    """Detect missing candles in a uniformly-spaced OHLCV DataFrame, insert
    missing timestamps, and forward-fill gaps of ≤ *max_ffill_candles* bars.
    Gaps larger than that are filled but a WARNING is logged.

    Parameters
    ----------
    df:
        DataFrame with a ``DatetimeIndex`` (UTC) and columns open/high/low/close/volume.
        Must already be sorted ascending.
    freq:
        Pandas offset alias for the expected candle frequency, e.g. ``"1min"``, ``"5min"``.
    max_ffill_candles:
        Maximum consecutive missing bars that are silently forward-filled.
        Larger gaps are still filled but trigger a WARNING.
    symbol:
        Optional label used in log messages.

    Returns
    -------
    pd.DataFrame
        Reindexed, gap-filled DataFrame.  A boolean column ``is_filled`` marks
        synthetic rows.
    """
    if df.empty:
        return df

    # Build a complete date range at the target frequency
    full_idx = pd.date_range(
        start=df.index[0],
        end=df.index[-1],
        freq=freq,
        tz="UTC",
    )

    missing = full_idx.difference(df.index)
    if len(missing) == 0:
        df = df.copy()
        df["is_filled"] = False
        return df

    logger.debug(
        "[{}] {} missing candles out of {} total before fill.",
        symbol or freq, len(missing), len(full_idx),
    )

    # Identify consecutive run lengths within *missing* so we can warn on long gaps
    _find_gap_runs(missing, freq=freq, max_ffill_candles=max_ffill_candles, symbol=symbol)

    # Reindex to the full grid; NaNs appear for missing bars
    df = df.reindex(full_idx)
    df["is_filled"] = df["close"].isna()

    # Forward-fill OHLCV (volume = 0 for synthetic bars is more honest)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)

    return df


def _find_gap_runs(
    missing: pd.DatetimeIndex,
    freq: str,
    max_ffill_candles: int,
    symbol: str,
) -> None:
    """Log warnings for missing runs that exceed *max_ffill_candles*."""
    if len(missing) == 0:
        return

    td = pd.tseries.frequencies.to_offset(freq)
    assert td is not None

    run_start = missing[0]
    run_len = 1
    for i in range(1, len(missing)):
        if missing[i] - missing[i - 1] == td:
            run_len += 1
        else:
            if run_len > max_ffill_candles:
                logger.warning(
                    "[{}] Large gap of {} candles starting at {} (limit={}).",
                    symbol or freq, run_len, run_start, max_ffill_candles,
                )
            run_start = missing[i]
            run_len = 1
    # flush last run
    if run_len > max_ffill_candles:
        logger.warning(
            "[{}] Large gap of {} candles starting at {} (limit={}).",
            symbol or freq, run_len, run_start, max_ffill_candles,
        )


# ── Misc helpers ──────────────────────────────────────────────────────────────

def timeframe_to_pandas_freq(timeframe: str) -> str:
    """Convert a CCXT/Alpaca timeframe string to a pandas offset alias.

    Examples
    --------
    >>> timeframe_to_pandas_freq("1m")
    '1min'
    >>> timeframe_to_pandas_freq("5m")
    '5min'
    >>> timeframe_to_pandas_freq("1h")
    '1h'
    """
    mapping: dict[str, str] = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1D",
    }
    result = mapping.get(timeframe)
    if result is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Supported: {list(mapping)}")
    return result


def chunk_date_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: pd.Timedelta,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) into non-overlapping chunks of *chunk_size*.

    Parameters
    ----------
    start, end:
        Inclusive start, exclusive end.
    chunk_size:
        Width of each chunk.

    Returns
    -------
    list of (chunk_start, chunk_end) pairs, all in UTC.
    """
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk_size, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks
