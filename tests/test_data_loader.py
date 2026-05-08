"""
tests/test_data_loader.py
=========================
Phase 2 DataStore tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.data_loader import (
    DataStore,
    get_latest_atr,
    get_latest_hurst,
)

pytestmark = pytest.mark.phase2


def test_data_store_loads_existing_files(synthetic_parquet_dir):
    """DataStore.load_all() populates the cache for both BTC and TSLA."""
    store = DataStore(data_dir=synthetic_parquet_dir)
    store.load_all()
    assert "BTC" in store.loaded_symbols
    assert "TSLA" in store.loaded_symbols


def test_data_store_skips_missing(tmp_path):
    """A missing parquet does not raise; the symbol is simply not cached."""
    (tmp_path / "raw").mkdir()
    store = DataStore(data_dir=tmp_path / "raw")
    store.load_all()
    assert store.loaded_symbols == []


def test_refresh_status(synthetic_parquet_dir, tmp_path):
    """refresh() returns 'ok' for present files."""
    store = DataStore(data_dir=synthetic_parquet_dir)
    store.load_all()
    status = store.refresh()
    assert status["BTC"] == "ok"
    assert status["TSLA"] == "ok"


def test_refresh_missing_returns_missing(tmp_path):
    """refresh() returns 'missing' for symbols never loaded."""
    (tmp_path / "raw").mkdir()
    store = DataStore(data_dir=tmp_path / "raw")
    status = store.refresh()
    assert all(v == "missing" for v in status.values())


def test_get_dataframe_returns_none_for_unloaded(tmp_path):
    """get_dataframe() returns None when nothing is cached."""
    (tmp_path / "raw").mkdir()
    store = DataStore(data_dir=tmp_path / "raw")
    assert store.get_dataframe("BTC") is None


def test_is_supported():
    """Known symbols return True; unknown return False."""
    store = DataStore()
    assert store.is_supported("BTC") is True
    assert store.is_supported("DOGE") is False


def test_get_latest_atr_returns_last_non_nan():
    """get_latest_atr ignores trailing NaN values."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"atr_14": [1.0, 2.0, 3.0, np.nan, np.nan]}, index=idx)
    assert get_latest_atr(df) == pytest.approx(3.0)


def test_get_latest_hurst_returns_last_non_nan():
    """get_latest_hurst ignores trailing NaN values."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"hurst_dfa": [np.nan, 0.4, 0.42, 0.41, np.nan]}, index=idx)
    assert get_latest_hurst(df) == pytest.approx(0.41)


def test_get_latest_atr_missing_column_raises():
    """get_latest_atr raises if column is absent."""
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="atr_14"):
        get_latest_atr(df)


def test_get_latest_hurst_all_nan_raises():
    """get_latest_hurst raises if all values are NaN."""
    df = pd.DataFrame({"hurst_dfa": [np.nan, np.nan]})
    with pytest.raises(ValueError):
        get_latest_hurst(df)
