"""
tests/test_smoke.py
===================
End-to-end smoke test: synthesize 5,000 bars, compute features, run backtest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_engineer import (
    compute_atr,
    compute_bollinger_bands,
    compute_sma,
)
from pipeline.storage import load_parquet, save_parquet
from backtest import simulator as sim_module
from backtest.simulator import simulate_trades

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def _build_synthetic(n_bars: int, seed: int = 7) -> pd.DataFrame:
    """OU-style mean-reverting series, 5-minute bars."""
    rng = np.random.default_rng(seed)
    theta, mu, sigma = 0.10, 100.0, 0.4
    x = np.empty(n_bars)
    x[0] = mu
    for i in range(1, n_bars):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()
    x = np.maximum(x, 1.0)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="5min", tz="UTC", name="timestamp")
    return pd.DataFrame({
        "open": x, "high": x + 0.2, "low": x - 0.2, "close": x,
        "volume": rng.uniform(100, 1000, n_bars),
    }, index=idx)


def test_smoke_pipeline_to_backtest(tmp_path: Path, monkeypatch):
    """5,000 bar synthetic series flows through features, parquet, backtest."""
    # Force a constant Kelly fraction so the textbook (p, b) sizer doesn't
    # zero out every trade (this is a known conservative behaviour in the
    # production sizer for OTM puts).
    monkeypatch.setattr(sim_module, "compute_kelly_fraction",
                        lambda premium, strike, delta_target: 0.05)
    df = _build_synthetic(5_000)

    df["atr_14"] = compute_atr(df, 14)
    df["sma_200"] = compute_sma(df, 200)
    bb = compute_bollinger_bands(df, 20, 2.0)
    df = pd.concat([df, bb], axis=1)
    # Synthetic Hurst — well below 0.45 to mark mean-reverting
    df["hurst_dfa"] = 0.30

    save_parquet(df, "TEST_SMOKE", "5m", base_dir=tmp_path)
    loaded = load_parquet("TEST_SMOKE", "5m", base_dir=tmp_path)
    assert loaded is not None
    assert len(loaded) == 5_000

    # Backtester expects daily-ish frames — feed it the loaded frame
    trades, equity = simulate_trades(
        loaded, symbol="TEST", initial_portfolio=100_000.0,
    )
    # We expect *some* trades to fire on a mean-reverting series
    assert len(trades) > 0, "Smoke backtest produced zero trades — entry conditions never met"
    assert len(equity) == len(loaded)
    assert np.isfinite(equity.iloc[-1])


def test_smoke_classifier_on_meanreverting(synthetic_ohlcv, mock_chronos_forecaster):
    """Classifier with narrow Chronos spread + low Hurst → MEAN_REVERTING."""
    from regime.classifier import classify_regime, Regime

    df = synthetic_ohlcv(n_bars=600, regime="meanreverting", seed=11)
    forecast = mock_chronos_forecaster.predict(
        context=df["close"].values, prediction_length=10,
    )
    res = classify_regime(
        q10=forecast.q10,
        q90=forecast.q90,
        current_atr=1.0,         # large ATR → narrow spread relative
        hurst_value=0.30,
    )
    assert res.regime == Regime.MEAN_REVERTING
    assert res.confidence >= 0.85
