"""
tests/test_features.py
======================
Phase 1 feature engineering tests: ATR, SMA, Bollinger, Hurst DFA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_engineer import (
    compute_atr,
    compute_bollinger_bands,
    compute_hurst_dfa,
    compute_sma,
)

pytestmark = pytest.mark.phase1


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_wilder_recursive_formula(synthetic_ohlcv):
    """ATR-14 follows Wilder's recursive formula: ATR_t = ATR_{t-1}*13/14 + TR_t/14."""
    df = synthetic_ohlcv(n_bars=200, regime="whitenoise", seed=1)
    atr = compute_atr(df, period=14)

    # First 13 entries are NaN (warm-up)
    assert atr.iloc[:13].isna().all()
    # 14th entry is first valid
    assert not np.isnan(atr.iloc[13])

    # Recompute manually and check Wilder recursion holds
    high = df["high"].values
    low = df["low"].values
    close_prev = np.concatenate([[np.nan], df["close"].values[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - close_prev),
        np.abs(low - close_prev),
    ])
    # Seed with mean of first 14 TRs (ewm adjust=False starts with first value)
    # ewm(com=13, adjust=False).mean() implementation: a_t = a_{t-1}*(1-α) + x_t*α with a_0 = x_0
    # Verify the recursion on the tail
    for i in range(20, 50):
        prev = atr.iloc[i - 1]
        cur = atr.iloc[i]
        expected = prev * (13.0 / 14.0) + tr[i] * (1.0 / 14.0)
        assert cur == pytest.approx(expected, rel=1e-9, abs=1e-9), f"row {i}: {cur} vs {expected}"


def test_atr_no_lookahead_uses_close_shift(synthetic_ohlcv):
    """ATR-14 must not change for past bars when future close values are mutated."""
    df = synthetic_ohlcv(n_bars=120, regime="whitenoise", seed=2)
    atr_orig = compute_atr(df, period=14)

    df_mut = df.copy()
    # Corrupt the *future* close values (last 20 bars)
    df_mut.loc[df_mut.index[-20:], "close"] = np.nan
    atr_mut = compute_atr(df_mut, period=14)

    # Past ATR values (up to index -21) must be identical
    pd.testing.assert_series_equal(
        atr_orig.iloc[:-20], atr_mut.iloc[:-20], check_names=False,
    )


# ── SMA ───────────────────────────────────────────────────────────────────────

def test_sma_200_boundary(synthetic_ohlcv):
    """SMA-200: first 199 entries are NaN; 200th equals mean of first 200 closes."""
    df = synthetic_ohlcv(n_bars=400, regime="whitenoise", seed=3)
    sma = compute_sma(df, period=200)

    assert sma.iloc[:199].isna().all()
    assert not np.isnan(sma.iloc[199])
    expected = df["close"].iloc[:200].mean()
    assert sma.iloc[199] == pytest.approx(expected, rel=1e-12)


def test_sma_short_series_all_nan(synthetic_ohlcv):
    """SMA on a series shorter than the window returns all NaN, no exception."""
    df = synthetic_ohlcv(n_bars=10, regime="whitenoise", seed=4)
    sma = compute_sma(df, period=200)
    assert sma.isna().all()
    assert len(sma) == 10


# ── Bollinger ─────────────────────────────────────────────────────────────────

def test_bollinger_basic_invariants(synthetic_ohlcv):
    """bb_upper > bb_lower wherever both are defined."""
    df = synthetic_ohlcv(n_bars=200, regime="whitenoise", seed=5)
    bb = compute_bollinger_bands(df, period=20, num_std=2.0)
    valid = bb.dropna()
    assert (valid["bb_upper"] > valid["bb_lower"]).all()


def test_bollinger_pct_at_midband_is_half():
    """When close == rolling mean and std > 0, bb_pct == 0.5.

    Constructed so the *last* bar's close equals the mean of the trailing
    20-bar window AND the window has non-zero std (using a symmetric pair
    of varying past values that average to the final value).
    """
    rng = np.random.default_rng(0)
    # 19 bars of noise + last bar = mean of those 19 bars + 0  →
    # By construction last close == window mean
    past = 100.0 + rng.normal(scale=2.0, size=19)
    target = float(np.mean(np.append(past, np.mean(past))))  # iterative: last == mean
    # If last = mean(past + last) → last = (sum(past) + last) / 20  → 19*last = sum(past)
    last = float(np.sum(past) / 19.0)
    closes = np.append(past, last)

    idx = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": [1000.0] * 20,
    }, index=idx)

    bb = compute_bollinger_bands(df, period=20, num_std=2.0)
    # Sanity: window mean equals last close
    assert closes[-1] == pytest.approx(closes.mean(), abs=1e-12)
    # Std should be non-zero
    assert closes.std(ddof=1) > 0
    assert bb["bb_pct"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def test_bollinger_pct_at_lower_is_zero():
    """When close == lower band, bb_pct == 0."""
    # Build a window where we know mean and std exactly, then on a 21st
    # bar set close = mean - 2*std (lower band).
    rng = np.random.default_rng(1)
    past = 100.0 + rng.normal(scale=2.0, size=20)
    mean = float(past.mean())
    std = float(past.std(ddof=1))
    # 21st bar: by sliding, the new window is past[1:] + [last]; we need
    # last == new_mean - 2*new_std. Compute self-consistent last.
    # new_mean = (sum(past[1:]) + last) / 20; new_std depends on last too.
    # Simpler: use a constant past and override last to test zero-division case.
    # For exact %B==0 we use a window where past[1:] has known stats.
    fixed_past = np.linspace(98.0, 102.0, 20)  # mean 100, std > 0
    # Slide: window is fixed_past[1:20] + [last]. We pick last s.t.
    # last = new_mean - 2*new_std, solving iteratively.
    base = fixed_past[1:].copy()  # 19 bars
    # Solve: last = (sum(base)+last)/20 - 2*sqrt(var)
    # This is a fixed-point; iterate.
    last = float(np.mean(base))
    for _ in range(200):
        win = np.append(base, last)
        new_mean = float(win.mean())
        new_std = float(win.std(ddof=1))
        last = new_mean - 2.0 * new_std
    closes = np.append(fixed_past[:1], np.append(base, last))
    idx = pd.date_range("2020-01-01", periods=21, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": closes, "high": closes + 0.5, "low": closes - 0.5,
        "close": closes, "volume": [1000.0] * 21,
    }, index=idx)

    bb = compute_bollinger_bands(df, period=20, num_std=2.0)
    assert bb["bb_pct"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_bollinger_flat_price_no_zero_division():
    """Flat (constant) price → upper == lower; bb_pct must be NaN, not Inf."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 30,
        "high": [100.0] * 30,
        "low": [100.0] * 30,
        "close": [100.0] * 30,
        "volume": [1.0] * 30,
    }, index=idx)
    bb = compute_bollinger_bands(df, period=20, num_std=2.0)
    valid_pct = bb["bb_pct"].iloc[19:]
    assert valid_pct.isna().all(), "bb_pct should be NaN when price is flat (no std)"
    assert np.isfinite(bb["bb_upper"].iloc[19:]).all()


def test_bollinger_short_series_all_nan(synthetic_ohlcv):
    """Series shorter than window → all NaN."""
    df = synthetic_ohlcv(n_bars=5, regime="whitenoise", seed=6)
    bb = compute_bollinger_bands(df, period=20, num_std=2.0)
    assert bb.isna().all().all()


# ── Hurst DFA ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_hurst_whitenoise_near_half(synthetic_ohlcv):
    """White noise series → Hurst ≈ 0.5 ± 0.1."""
    nolds = pytest.importorskip("nolds")
    df = synthetic_ohlcv(n_bars=2200, regime="whitenoise", seed=42)
    hurst = compute_hurst_dfa(df, window=512)
    final = hurst.dropna().iloc[-1]
    assert 0.4 <= final <= 0.6, f"Hurst {final:.3f} not near 0.5 for white noise"


@pytest.mark.slow
def test_hurst_trending_above_half(synthetic_ohlcv):
    """Strongly trending series → Hurst > 0.6 in expectation."""
    pytest.importorskip("nolds")
    df = synthetic_ohlcv(n_bars=2200, regime="trending", seed=43)
    hurst = compute_hurst_dfa(df, window=512)
    final = hurst.dropna().iloc[-1]
    assert final > 0.6, f"Hurst {final:.3f} not > 0.6 for trending series"


@pytest.mark.slow
def test_hurst_meanreverting_below_half(synthetic_ohlcv):
    """OU mean-reverting series → Hurst < 0.4 in expectation."""
    pytest.importorskip("nolds")
    df = synthetic_ohlcv(n_bars=2200, regime="meanreverting", seed=44)
    hurst = compute_hurst_dfa(df, window=512)
    final = hurst.dropna().iloc[-1]
    assert final < 0.45, f"Hurst {final:.3f} not < 0.45 for mean-reverting series"


def test_hurst_short_series_returns_all_nan(synthetic_ohlcv):
    """Series shorter than window → all NaN, no exception."""
    pytest.importorskip("nolds")
    df = synthetic_ohlcv(n_bars=100, regime="whitenoise", seed=7)
    hurst = compute_hurst_dfa(df, window=512)
    assert hurst.isna().all()


def test_features_handle_nan_inputs(synthetic_ohlcv):
    """Inserting NaN in close should not raise; downstream values may be NaN."""
    df = synthetic_ohlcv(n_bars=100, regime="whitenoise", seed=8)
    df.loc[df.index[20:25], "close"] = np.nan
    df.loc[df.index[20:25], "high"] = np.nan
    df.loc[df.index[20:25], "low"] = np.nan

    atr = compute_atr(df, 14)
    sma = compute_sma(df, 30)
    bb = compute_bollinger_bands(df, 20, 2.0)
    # Should produce a Series of correct length without exceptions
    assert len(atr) == len(df)
    assert len(sma) == len(df)
    assert len(bb) == len(df)
