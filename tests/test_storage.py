"""
tests/test_storage.py
=====================
Phase 1 storage tests: parquet round-trip, snappy compression, missing-file handling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pipeline.storage import (
    load_parquet,
    merge_and_deduplicate,
    raw_path,
    save_parquet,
)

pytestmark = pytest.mark.phase1


def _build_df(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC", name="timestamp")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "open": rng.uniform(99, 101, n),
        "high": rng.uniform(101, 102, n),
        "low": rng.uniform(98, 99, n),
        "close": rng.uniform(99, 101, n),
        "volume": rng.uniform(0, 1000, n),
    }, index=idx)


def test_parquet_round_trip(tmp_path: Path):
    """save → load returns an identical DataFrame including DatetimeIndex."""
    df = _build_df(60)
    save_parquet(df, "BTC_USD", "1m", base_dir=tmp_path)
    loaded = load_parquet("BTC_USD", "1m", base_dir=tmp_path)
    assert loaded is not None
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert str(loaded.index.tz) == "UTC"
    pd.testing.assert_frame_equal(
        df, loaded, check_freq=False, check_index_type=False
    )


def test_parquet_uses_snappy_compression(tmp_path: Path):
    """Inspect the parquet metadata and confirm Snappy compression."""
    df = _build_df(20)
    save_parquet(df, "BTC_USD", "1m", base_dir=tmp_path)
    path = raw_path("BTC_USD", "1m", base_dir=tmp_path)
    pqfile = pq.ParquetFile(str(path))
    # Pull the column compression from the first row group
    rg = pqfile.metadata.row_group(0)
    compressions = {rg.column(i).compression for i in range(rg.num_columns)}
    assert "SNAPPY" in compressions, f"Expected SNAPPY, got {compressions}"


def test_load_parquet_missing_returns_none(tmp_path: Path):
    """Loading a non-existent symbol returns None (no exception)."""
    out = load_parquet("ZZZ", "1m", base_dir=tmp_path)
    assert out is None


def test_save_localizes_naive_index(tmp_path: Path):
    """A tz-naive DatetimeIndex is localised to UTC on save."""
    df = _build_df(10)
    df = df.copy()
    df.index = df.index.tz_localize(None)  # strip tz
    save_parquet(df, "BTC_USD", "1m", base_dir=tmp_path)
    loaded = load_parquet("BTC_USD", "1m", base_dir=tmp_path)
    assert loaded is not None
    assert str(loaded.index.tz) == "UTC"


def test_merge_and_deduplicate_prefers_new():
    """When `existing` and `new` overlap on timestamps, `new` wins."""
    idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC", name="timestamp")
    existing = pd.DataFrame({
        "open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5], "close": [1, 2, 3, 4, 5],
        "volume": [10, 20, 30, 40, 50],
    }, index=idx, dtype=float)
    new = pd.DataFrame({
        "open": [99, 99], "high": [99, 99], "low": [99, 99],
        "close": [99, 99], "volume": [0, 0],
    }, index=idx[3:], dtype=float)

    merged = merge_and_deduplicate(existing, new)
    assert len(merged) == 5
    # Last two rows should reflect 'new'
    assert merged["close"].iloc[-1] == 99.0
    assert merged["close"].iloc[-2] == 99.0
    # First three are unchanged
    assert merged["close"].iloc[0] == 1.0
    assert merged["close"].iloc[2] == 3.0
