"""
tests/test_options_math.py
==========================
Phase 3 Black-Scholes pricing, delta, strike solver and realized vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from backtest.options_math import (
    bs_put_delta,
    bs_put_price,
    compute_realized_vol,
    solve_strike_for_delta,
)

pytestmark = pytest.mark.phase3


# ── Black-Scholes pricing ─────────────────────────────────────────────────────

def test_bs_put_price_published_value():
    """ATM put, S=100, K=100, r=0.05, σ=0.20, T=1 → ≈ $5.5735."""
    p = bs_put_price(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
    assert p == pytest.approx(5.5735, abs=0.01)


def test_bs_put_price_deep_itm_floor():
    """Deep ITM put (S << K) is at least intrinsic value."""
    p = bs_put_price(S=50, K=100, r=0.05, sigma=0.2, T=1.0)
    intrinsic = 100 * np.exp(-0.05 * 1.0) - 50
    # BS value should be ≥ discounted intrinsic, but always ≥ 0
    assert p >= max(intrinsic, 0.0) - 1e-6


def test_bs_put_delta_formula_and_range():
    """Δ_put = N(d1) - 1, in (-1, 0]."""
    S, K, r, sigma, T = 100, 105, 0.05, 0.2, 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    expected = norm.cdf(d1) - 1.0
    actual = bs_put_delta(S=S, K=K, r=r, sigma=sigma, T=T)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert -1.0 < actual <= 0.0


def test_bs_put_delta_atm_near_minus_half():
    """ATM put delta is approximately -0.5 (slightly higher with positive r)."""
    d = bs_put_delta(S=100, K=100, r=0.0, sigma=0.2, T=1.0)
    assert d == pytest.approx(-0.5, abs=0.05)


# ── Strike solver ─────────────────────────────────────────────────────────────

def test_strike_solver_round_trip():
    """Solve K for target delta, recompute delta from K — must match."""
    S, r, sigma, T = 100.0, 0.05, 0.25, 30 / 365.0
    target = -0.20
    K = solve_strike_for_delta(S=S, r=r, sigma=sigma, T=T, target_delta=target)
    actual_delta = bs_put_delta(S=S, K=K, r=r, sigma=sigma, T=T)
    assert actual_delta == pytest.approx(target, abs=1e-4)
    # 20-delta put should be OTM (K < S)
    assert K < S


def test_strike_solver_deep_otm_returns_finite():
    """Very small target delta still produces a finite strike via fallback."""
    S = 100.0
    K = solve_strike_for_delta(S=S, r=0.05, sigma=0.20, T=30 / 365.0, target_delta=-0.001)
    assert np.isfinite(K)
    assert 0 < K < S


def test_strike_solver_degenerate_inputs_no_crash():
    """sigma=0, T<=0, S<=0 → fallback returns finite values, no exception."""
    K = solve_strike_for_delta(S=100, r=0.05, sigma=0.0, T=1.0, target_delta=-0.2)
    assert np.isfinite(K)
    K = solve_strike_for_delta(S=100, r=0.05, sigma=0.2, T=-1.0, target_delta=-0.2)
    assert np.isfinite(K)
    K = solve_strike_for_delta(S=0.0, r=0.05, sigma=0.2, T=1.0, target_delta=-0.2)
    assert np.isfinite(K)


def test_bs_put_price_zero_sigma_returns_intrinsic():
    """sigma=0 returns max(K-S, 0); should not crash."""
    assert bs_put_price(S=100, K=110, r=0.05, sigma=0.0, T=1.0) == pytest.approx(10.0)
    assert bs_put_price(S=120, K=110, r=0.05, sigma=0.0, T=1.0) == pytest.approx(0.0)


# ── Realized volatility ───────────────────────────────────────────────────────

def test_realized_vol_known_sigma():
    """Synthetic GBM with known annualized σ → realized_vol within tolerance."""
    rng = np.random.default_rng(42)
    n = 1000
    sigma_true = 0.30
    daily_sigma = sigma_true / np.sqrt(252)
    rets = rng.normal(0.0, daily_sigma, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    s = pd.Series(closes, index=idx)
    rv = compute_realized_vol(s, window=60, trading_days=252).dropna()
    # Mean realized vol over the series should be close to the true sigma
    mean_rv = rv.mean()
    assert mean_rv == pytest.approx(sigma_true, rel=0.15)


def test_realized_vol_warmup_nan(synthetic_ohlcv):
    """First `window` values are NaN."""
    df = synthetic_ohlcv(n_bars=200, regime="whitenoise", seed=0)
    rv = compute_realized_vol(df["close"], window=30)
    assert rv.iloc[:30].isna().all()
    assert not np.isnan(rv.iloc[30])
