"""
tests/test_pipeline_utils.py
============================
Phase 1 utility tests: gap detection / forward-fill, rate limiter, retry decorator.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from pipeline.utils import (
    AsyncRateLimiter,
    async_retry,
    chunk_date_range,
    detect_and_fill_gaps,
    timeframe_to_pandas_freq,
)

pytestmark = pytest.mark.phase1


def _make_minute_df(times: list[pd.Timestamp]) -> pd.DataFrame:
    """Build an OHLCV DataFrame from explicit timestamps with sequential closes."""
    n = len(times)
    closes = np.linspace(100.0, 100.0 + n - 1, n)
    df = pd.DataFrame({
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.full(n, 10.0),
    }, index=pd.DatetimeIndex(times, tz="UTC", name="timestamp"))
    return df


def test_gap_detection_inserts_missing_timestamps():
    """A 2-bar hole in a 1-minute series is filled in and marked is_filled."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    # bars at 00, 01, 04, 05  → missing 02, 03
    times = [base, base + pd.Timedelta("1min"),
             base + pd.Timedelta("4min"), base + pd.Timedelta("5min")]
    df = _make_minute_df(times)
    out = detect_and_fill_gaps(df, freq="1min", max_ffill_candles=5)
    assert len(out) == 6
    # The synthetic rows (idx 2 and 3) should be marked is_filled
    assert out["is_filled"].iloc[2] is np.True_ or bool(out["is_filled"].iloc[2])
    assert bool(out["is_filled"].iloc[3])
    # Forward-fill: closes at 02 == close at 01
    assert out["close"].iloc[2] == out["close"].iloc[1]
    # Volume zeroed for synthetic
    assert out["volume"].iloc[2] == 0.0


def test_gap_small_silent_no_warning(caplog):
    """A 2-bar gap (≤ max_ffill_candles=5) does NOT log a WARNING."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    times = [base, base + pd.Timedelta("1min"),
             base + pd.Timedelta("4min"), base + pd.Timedelta("5min")]
    df = _make_minute_df(times)

    # Capture loguru by routing to caplog handler
    sink_id = logger.add(lambda m: caplog.records.append(m), level="WARNING")
    try:
        detect_and_fill_gaps(df, freq="1min", max_ffill_candles=5)
    finally:
        logger.remove(sink_id)

    # No WARNING messages logged
    warning_msgs = [str(r) for r in caplog.records if "Large gap" in str(r)]
    assert warning_msgs == []


def test_gap_large_logs_warning():
    """A 7-bar gap (> max_ffill_candles=5) DOES log a WARNING."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    # bars at 00, 01, 09, 10  → missing 7 bars (02..08)
    times = [base, base + pd.Timedelta("1min"),
             base + pd.Timedelta("9min"), base + pd.Timedelta("10min")]
    df = _make_minute_df(times)

    seen = []
    sink_id = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        detect_and_fill_gaps(df, freq="1min", max_ffill_candles=5)
    finally:
        logger.remove(sink_id)

    assert any("Large gap" in s for s in seen), f"Expected 'Large gap' warning, got: {seen}"


def test_token_bucket_rate_limits():
    """Acquiring more tokens than capacity must take measurable time."""
    async def _run():
        # 5 tokens/sec, capacity 2 → 5 tokens total takes ~0.6s
        rl = AsyncRateLimiter(rate=5.0, period=1.0, burst=2.0)
        start = time.monotonic()
        for _ in range(5):
            await rl.acquire(1.0)
        elapsed = time.monotonic() - start
        return elapsed

    elapsed = asyncio.run(_run())
    # Burst 2 free, then 3 more @ 5/s ≈ 0.6s
    assert elapsed >= 0.5, f"Rate limiter too fast: {elapsed:.3f}s"
    assert elapsed < 2.0, f"Rate limiter unexpectedly slow: {elapsed:.3f}s"


def test_async_retry_exponential_growth():
    """Verify the decorator retries with growing delay and eventually succeeds."""
    call_log = []

    @async_retry(max_attempts=4, base_delay=0.01, max_delay=0.05, jitter=0.0,
                 retryable_exceptions=(RuntimeError,))
    async def flaky():
        call_log.append(time.monotonic())
        if len(call_log) < 3:
            raise RuntimeError("simulated")
        return "ok"

    result = asyncio.run(flaky())
    assert result == "ok"
    assert len(call_log) == 3
    # Gaps should grow: g2 >= g1
    g1 = call_log[1] - call_log[0]
    g2 = call_log[2] - call_log[1]
    assert g2 >= g1 - 0.005, f"Backoff did not grow: g1={g1:.4f} g2={g2:.4f}"


def test_async_retry_propagates_after_exhaustion():
    """When all attempts fail, the original exception is raised."""

    @async_retry(max_attempts=2, base_delay=0.001, max_delay=0.005, jitter=0.0,
                 retryable_exceptions=(ValueError,))
    async def always_fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(always_fail())


def test_timeframe_to_pandas_freq():
    """Mapping table is correct for common timeframes."""
    assert timeframe_to_pandas_freq("1m") == "1min"
    assert timeframe_to_pandas_freq("5m") == "5min"
    assert timeframe_to_pandas_freq("1h") == "1h"
    with pytest.raises(ValueError):
        timeframe_to_pandas_freq("99x")


def test_chunk_date_range_no_overlap_or_gap():
    """chunk_date_range produces contiguous, non-overlapping slices."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-10", tz="UTC")
    chunks = chunk_date_range(start, end, pd.Timedelta(days=3))
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # Each chunk's end == next chunk's start
    for (a_s, a_e), (b_s, _) in zip(chunks, chunks[1:]):
        assert a_e == b_s
