"""
tests/test_simulator.py
=======================
Phase 3 simulator tests: trade outcomes, equity curve, portfolio combination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import simulator as sim_module
from backtest.simulator import (
    INITIAL_PORTFOLIO,
    TradeRecord,
    combine_equity_curves,
    simulate_trades,
    trades_to_dataframe,
)

pytestmark = pytest.mark.phase3


@pytest.fixture
def force_kelly(monkeypatch):
    """Override the Kelly sizer to a constant 10% so trades fire deterministically.

    The textbook (p, b) Kelly used in production is intentionally very
    conservative for OTM puts (returns 0 across all realistic σ).  These
    tests focus on the simulator's *trade execution and P&L* paths, which we
    isolate by forcing a positive Kelly fraction.
    """
    monkeypatch.setattr(sim_module, "compute_kelly_fraction",
                        lambda premium, strike, delta_target: 0.10)
    return 0.10


def _build_df_with_one_entry(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Build a daily OHLCV+features DataFrame engineered to fire exactly one entry.

    The close path needs nonzero realized vol (≥ 30 bars of variation) for the
    Black-Scholes pipeline to produce a non-zero premium and Kelly fraction.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    # Moderate daily vol — Kelly is force-overridden in tests, so we can use
    # realistic values that produce a strike near the spot price.
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": np.full(n, 1.0),
        "atr_14": np.full(n, 1.0),
        "sma_200": np.full(n, 50.0),       # close > sma always (close hovers near 100)
        "bb_pct": np.full(n, 0.40),        # default: no entry, no exit
        "hurst_dfa": np.full(n, 0.40),     # mean-reverting
    }, index=idx)
    # Mark exactly one entry trigger at bar 35 (signal shifts to bar 36)
    df.loc[idx[35], "bb_pct"] = 0.05
    return df


def test_profitable_put_keeps_premium(force_kelly):
    """Underlying flat → put expires worthless → P&L positive (≈ premium ratio)."""
    df = _build_df_with_one_entry()
    trades, equity = simulate_trades(df, symbol="TEST", initial_portfolio=100_000.0)
    assert len(trades) >= 1, "expected at least one trade"
    t = trades[0]
    # Flat price means exit_price ≈ entry → put expires worthless → pnl > 0
    assert t.pnl > 0, f"Expected profit, got pnl={t.pnl}"
    assert t.exit_reason == "expiry"


def test_losing_put_assigned(force_kelly):
    """If close < strike at exit → loss = strike - close - premium received."""
    df = _build_df_with_one_entry()
    # Crash price after entry (bar 36 onward) so put goes deep ITM
    # Crash AFTER entry (bar 37 onward).  Entry is at bar 36 (close ~100);
    # strike will be ~98.  Then close drops to 30 → put deep ITM at expiry.
    df.loc[df.index[37:], "close"] = 30.0
    df.loc[df.index[37:], "bb_pct"] = 0.30  # avoid retroactive exit signals
    df.loc[df.index[37:], "sma_200"] = 20.0  # keep close > sma_200 to avoid trend-break exit
    trades, _ = simulate_trades(df, symbol="TEST", initial_portfolio=100_000.0)
    assert len(trades) >= 1
    t = trades[0]
    # Strike < entry but >> 70 → put is ITM → loss
    assert t.pnl < 0, f"Expected loss, got pnl={t.pnl}"


def test_equity_curve_monotonic_length(force_kelly):
    """Equity curve has the same length as input frame."""
    df = _build_df_with_one_entry()
    _, equity = simulate_trades(df, symbol="TEST", initial_portfolio=100_000.0)
    assert len(equity) == len(df)
    assert not equity.isna().any(), "equity curve should be fully filled"


def test_combine_equity_curves_50_50():
    """Two flat equity curves combine to the initial portfolio value."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    eq_btc = pd.Series(100_000.0, index=idx)
    eq_tsla = pd.Series(100_000.0, index=idx)
    combined = combine_equity_curves(eq_btc, eq_tsla, initial_portfolio=100_000.0)
    # Each side normalised to half = 50_000; sum = 100_000 throughout
    assert combined.iloc[0] == pytest.approx(100_000.0)
    assert combined.iloc[-1] == pytest.approx(100_000.0)
    assert ((combined - 100_000.0).abs() < 1e-6).all()


def test_combine_equity_curves_proportional():
    """If BTC doubles and TSLA flat → portfolio = 0.5*200% + 0.5*100% = 150%."""
    idx = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    eq_btc = pd.Series([100_000.0, 200_000.0], index=idx)
    eq_tsla = pd.Series([100_000.0, 100_000.0], index=idx)
    combined = combine_equity_curves(eq_btc, eq_tsla, initial_portfolio=100_000.0)
    assert combined.iloc[0] == pytest.approx(100_000.0)
    assert combined.iloc[1] == pytest.approx(150_000.0)


def test_trades_to_dataframe_empty():
    """trades_to_dataframe([]) returns a DataFrame with the expected columns."""
    df = trades_to_dataframe([])
    assert df.empty
    for col in ("symbol", "entry_date", "exit_date", "pnl", "pnl_pct", "exit_reason"):
        assert col in df.columns


def test_trades_to_dataframe_records():
    """Conversion round-trips fields and sorts by entry_date."""
    t1 = TradeRecord(
        symbol="X", entry_date=pd.Timestamp("2020-01-02", tz="UTC"),
        exit_date=pd.Timestamp("2020-02-01", tz="UTC"),
        entry_price=100.0, exit_price=101.0, strike=95.0,
        premium=2.0, kelly_fraction=0.10, position_value=10_000.0,
        pnl=200.0, pnl_pct=0.02, exit_reason="expiry",
    )
    t0 = TradeRecord(
        symbol="X", entry_date=pd.Timestamp("2020-01-01", tz="UTC"),
        exit_date=pd.Timestamp("2020-01-30", tz="UTC"),
        entry_price=100.0, exit_price=99.0, strike=95.0,
        premium=2.0, kelly_fraction=0.10, position_value=10_000.0,
        pnl=180.0, pnl_pct=0.018, exit_reason="expiry",
    )
    df = trades_to_dataframe([t1, t0])
    assert df.iloc[0]["entry_date"] == pd.Timestamp("2020-01-01", tz="UTC")
    assert df.iloc[1]["pnl"] == pytest.approx(200.0)
