"""
tests/test_performance.py
=========================
Phase 3 performance metric tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.performance import (
    build_buyhold_benchmark,
    compute_calmar_ratio,
    compute_cagr,
    compute_max_drawdown,
    compute_metrics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_total_return,
    compute_win_rate,
)

pytestmark = pytest.mark.phase3


def _equity(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_total_return_simple():
    """Total return = (final - initial) / initial."""
    eq = _equity([100, 110, 120, 150])
    assert compute_total_return(eq) == pytest.approx(0.5)


def test_total_return_zero_start():
    """Initial == 0 returns 0 instead of dividing by zero."""
    eq = _equity([0, 100])
    assert compute_total_return(eq) == 0.0


def test_max_drawdown_known_series():
    """Peak 200 → trough 100 = -50% drawdown."""
    eq = _equity([100, 150, 200, 100, 120, 180])
    assert compute_max_drawdown(eq) == pytest.approx(-0.5, abs=1e-9)


def test_max_drawdown_no_drawdown():
    """Strictly increasing series → MDD == 0."""
    eq = _equity([100, 110, 120, 130])
    assert compute_max_drawdown(eq) == 0.0


def test_cagr_known():
    """100 → 121 over 2 years → CAGR ≈ 10%."""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"),
    ])
    eq = pd.Series([100.0, 121.0], index=idx)
    assert compute_cagr(eq) == pytest.approx(0.10, abs=0.005)


def test_sharpe_constant_returns_zero():
    """Constant returns → std=0 → Sharpe defined as 0 (avoid division by zero)."""
    eq = _equity([100] * 50)
    assert compute_sharpe_ratio(eq) == 0.0


def test_sharpe_positive_for_uptrend():
    """Steady positive daily returns produce positive Sharpe."""
    rng = np.random.default_rng(0)
    daily = 0.001 + rng.normal(0, 0.0001, 252)  # mean > 0, low noise
    eq = pd.Series(np.cumprod(1 + daily) * 100,
                   index=pd.date_range("2020-01-01", periods=252, freq="D", tz="UTC"))
    s = compute_sharpe_ratio(eq, risk_free_rate=0.0)
    assert s > 1.0


def test_sortino_handles_no_negatives():
    """If there are no negative returns, sortino returns 0 (no downside std)."""
    eq = _equity([100, 101, 102, 103, 104, 105])
    assert compute_sortino_ratio(eq) == 0.0


def test_calmar_ratio():
    """Calmar = CAGR / |MDD|."""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-07-01", tz="UTC"),
        pd.Timestamp("2021-01-01", tz="UTC"),
    ])
    eq = pd.Series([100.0, 50.0, 110.0], index=idx)
    cagr = compute_cagr(eq)
    mdd = compute_max_drawdown(eq)
    expected = cagr / abs(mdd)
    assert compute_calmar_ratio(eq) == pytest.approx(expected, abs=1e-9)


def test_win_rate():
    """Fraction of trades with pnl > 0."""
    df = pd.DataFrame({"pnl": [10, -5, 3, -2, 1], "pnl_pct": [0.01] * 5})
    assert compute_win_rate(df) == pytest.approx(0.6)


def test_metrics_empty_equity():
    """Empty equity produces zeros without raising."""
    eq = pd.Series(dtype=float)
    df = pd.DataFrame(columns=["pnl", "pnl_pct"])
    m = compute_metrics(eq, df, label="EMPTY")
    assert m.total_return == 0.0
    assert m.n_trades == 0


def test_buyhold_benchmark_total_return():
    """B&H final return = (final - initial)/initial when prices double."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    btc = pd.DataFrame({
        "open": [100.0] + [200.0] * 9,
        "close": np.linspace(100, 200, 10),
    }, index=idx)
    tsla = pd.DataFrame({
        "open": [50.0] + [100.0] * 9,
        "close": np.linspace(50, 100, 10),
    }, index=idx)
    eq = build_buyhold_benchmark(btc, tsla, start_date="2020-01-01",
                                 end_date="2030-01-01", initial_portfolio=100_000.0)
    # Both doubled → total return ≈ 100%
    assert compute_total_return(eq) == pytest.approx(1.0, rel=1e-3)
