"""
tests/conftest.py
=================
Shared pytest fixtures for the offline test suite.

Provides:
- `synthetic_ohlcv`: factory for OHLCV DataFrames with controllable regime
- `synthetic_parquet_dir`: a populated `data/raw/` style temp directory
- `mock_chronos_forecaster`: drop-in replacement for ChronosForecaster
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on sys.path so `pipeline.*`, `backtest.*`, `regime.*` import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep loguru quiet for tests unless explicitly debugged
try:
    from loguru import logger
    logger.remove()
    if os.environ.get("TEST_LOG"):
        logger.add(sys.stderr, level=os.environ.get("TEST_LOG", "INFO"))
except Exception:
    pass


Regime = Literal["whitenoise", "trending", "meanreverting"]


def _generate_ohlcv(
    n_bars: int,
    regime: Regime = "whitenoise",
    start: str = "2020-01-01",
    freq: str = "5min",
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV with controllable statistical properties.

    Parameters
    ----------
    regime:
        - 'whitenoise': geometric Brownian motion increments (Hurst ≈ 0.5)
        - 'trending':   biased random walk that drifts persistently up
        - 'meanreverting': Ornstein-Uhlenbeck around start_price
    """
    rng = np.random.default_rng(seed)
    if regime == "whitenoise":
        rets = rng.normal(loc=0.0, scale=0.005, size=n_bars)
        prices = start_price * np.exp(np.cumsum(rets))
    elif regime == "trending":
        # Strong positive autocorrelation: each return depends on prior return.
        # Persistence factor 0.6 gives a long-memory series where DFA on log-returns
        # detects H > 0.5.
        eps = rng.normal(0.0, 0.005, size=n_bars)
        rets = np.empty(n_bars)
        rets[0] = eps[0]
        for i in range(1, n_bars):
            rets[i] = 0.6 * rets[i - 1] + eps[i]
        prices = start_price * np.exp(np.cumsum(rets))
    elif regime == "meanreverting":
        # Strong negative autocorrelation in log-returns → anti-persistent.
        eps = rng.normal(0.0, 0.005, size=n_bars)
        rets = np.empty(n_bars)
        rets[0] = eps[0]
        for i in range(1, n_bars):
            rets[i] = -0.6 * rets[i - 1] + eps[i]
        prices = start_price * np.exp(np.cumsum(rets))
        prices = np.maximum(prices, 1.0)
    else:
        raise ValueError(f"Unknown regime '{regime}'.")

    # Build OHLC around the close path with small intrabar noise
    close = prices
    noise_high = np.abs(rng.normal(scale=0.0015, size=n_bars)) * close
    noise_low = np.abs(rng.normal(scale=0.0015, size=n_bars)) * close
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + noise_high
    low = np.minimum(open_, close) - noise_low
    volume = rng.integers(1_000, 100_000, size=n_bars).astype(float)

    idx = pd.date_range(start=start, periods=n_bars, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_ohlcv() -> Callable[..., pd.DataFrame]:
    """Factory fixture: call with kwargs to generate synthetic OHLCV.

    Example:
        df = synthetic_ohlcv(n_bars=600, regime='meanreverting', seed=7)
    """
    return _generate_ohlcv


@pytest.fixture
def synthetic_parquet_dir(tmp_path: Path, synthetic_ohlcv) -> Path:
    """Build a `data/raw/` directory with BTC and TSLA parquet files.

    Both files include core OHLCV plus the feature columns the regime API expects:
    atr_14, sma_200, bb_upper, bb_lower, bb_pct, hurst_dfa.
    """
    from pipeline.feature_engineer import (
        compute_atr,
        compute_bollinger_bands,
        compute_sma,
    )

    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    for sym, fname, regime in [
        ("BTC", "BTC_USD_5m.parquet", "meanreverting"),
        ("TSLA", "TSLA_5m.parquet", "trending"),
    ]:
        df = synthetic_ohlcv(n_bars=800, regime=regime, seed=hash(sym) % 1000)
        df["atr_14"] = compute_atr(df, 14)
        df["sma_200"] = compute_sma(df, 200)
        bb = compute_bollinger_bands(df, 20, 2.0)
        df = pd.concat([df, bb], axis=1)
        # Synthetic hurst: stable values per symbol
        df["hurst_dfa"] = 0.35 if regime == "meanreverting" else 0.65
        df.index.name = "timestamp"
        df.to_parquet(data_dir / fname, engine="pyarrow", compression="snappy", index=True)

    return data_dir


# ── Mock ChronosForecaster ────────────────────────────────────────────────────

@dataclass
class _MockForecast:
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    samples: np.ndarray
    prediction_length: int = 10

    @property
    def spread(self):
        return self.q90 - self.q10

    @property
    def mean_spread(self):
        return float(np.mean(self.spread))

    @property
    def forecast_low(self):
        return float(np.min(self.q10))

    @property
    def forecast_high(self):
        return float(np.max(self.q90))


class MockChronosForecaster:
    """Drop-in deterministic replacement for ChronosForecaster.

    `spread_factor` controls width relative to the last context value:
      - 0.005 → narrow band (MEAN_REVERTING signal)
      - 0.05  → wide band (TRENDING signal)
    """

    def __init__(self, spread_factor: float = 0.005, num_samples: int = 20):
        self._model_id = "mock/chronos"
        self._loaded = True
        self._spread_factor = spread_factor
        self._num_samples = num_samples

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:  # pragma: no cover
        self._loaded = True

    def predict(self, context: np.ndarray, prediction_length: int = 10):
        last = float(context[-1]) if len(context) > 0 else 100.0
        steps = np.arange(1, prediction_length + 1)
        q50 = np.full(prediction_length, last, dtype=np.float64)
        half = self._spread_factor * last
        q10 = q50 - half
        q90 = q50 + half
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=last, scale=half, size=(self._num_samples, prediction_length))
        return _MockForecast(q10=q10, q50=q50, q90=q90, samples=samples,
                             prediction_length=prediction_length)


@pytest.fixture
def mock_chronos_forecaster() -> MockChronosForecaster:
    """Narrow-spread mock forecaster (mean-reverting signal)."""
    return MockChronosForecaster(spread_factor=0.005)


@pytest.fixture
def mock_chronos_forecaster_trending() -> MockChronosForecaster:
    """Wide-spread mock forecaster (trending signal)."""
    return MockChronosForecaster(spread_factor=0.05)
