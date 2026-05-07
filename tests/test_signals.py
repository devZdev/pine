"""
tests/test_signals.py
=====================
Phase 3 entry/exit signal generator tests.

Confirms:
- All three entry conditions must hold simultaneously (boolean AND)
- Each condition can independently veto the entry
- Exit fires on EITHER take-profit (bb_pct > 0.5) OR trend-break (close < sma_200)
- All signals shift(1) — no look-ahead
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.signal_generator import (
    compute_atr_stop,
    generate_entry_signals,
    generate_exit_signals,
)

pytestmark = pytest.mark.phase3


def _frame(n: int) -> pd.DataFrame:
    """Build an empty dataframe with all columns required by signal_generator."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "close": np.full(n, 100.0),
        "sma_200": np.full(n, 90.0),
        "bb_pct": np.full(n, 0.10),     # below 0.20 by default
        "hurst_dfa": np.full(n, 0.40),  # below 0.45 by default
        "atr_14": np.full(n, 1.0),
    }, index=idx)


def test_entry_fires_when_all_conditions_true():
    """All three entry conditions met → entry True (after 1-bar shift)."""
    df = _frame(10)
    sig = generate_entry_signals(df, bb_pct_threshold=0.20, hurst_threshold=0.45)
    # Raw conditions are True everywhere; after shift(1) bars 1..9 are True, bar 0 False
    assert sig.iloc[0] == False  # noqa: E712
    assert sig.iloc[1:].all()


@pytest.mark.parametrize("col,bad_val", [
    ("bb_pct", 0.5),       # too high
    ("close", 80.0),       # below SMA
    ("hurst_dfa", 0.6),    # trending regime
])
def test_entry_blocked_if_any_condition_fails(col, bad_val):
    """Breaking any single condition disables the entry signal everywhere."""
    df = _frame(10)
    df[col] = bad_val
    sig = generate_entry_signals(df, bb_pct_threshold=0.20, hurst_threshold=0.45)
    assert not sig.any()


def test_exit_fires_on_take_profit():
    """bb_pct > 0.5 triggers exit (after shift)."""
    df = _frame(10)
    df["bb_pct"] = 0.70
    sig = generate_exit_signals(df)
    assert sig.iloc[1:].all()


def test_exit_fires_on_trend_break():
    """close < sma_200 triggers exit (after shift)."""
    df = _frame(10)
    df["close"] = 50.0
    sig = generate_exit_signals(df)
    assert sig.iloc[1:].all()


def test_exit_quiet_in_normal_state():
    """No exit fires when neither condition holds."""
    df = _frame(10)
    df["bb_pct"] = 0.20  # below 0.5 → no TP
    df["close"] = 100.0  # above sma 90 → no breakdown
    sig = generate_exit_signals(df)
    assert not sig.any()


def test_signals_no_lookahead():
    """A signal at bar t cannot have been emitted at bar t-1's close.

    Compute signal twice — once with full data, once with the future masked.
    Past signals must be identical.
    """
    df = _frame(20)
    # Set bars 10..19 to satisfy entry conditions only there
    df.loc[df.index[:10], "bb_pct"] = 0.5  # block entry
    sig_full = generate_entry_signals(df)

    df_past = df.copy()
    df_past.loc[df.index[15:], :] = np.nan
    sig_partial = generate_entry_signals(df_past)
    # Past signals (where data exists) should match
    pd.testing.assert_series_equal(
        sig_full.iloc[:14], sig_partial.iloc[:14], check_names=False,
    )


def test_atr_stop_triggers_below_threshold():
    """ATR stop activates when close drops below entry - 2*ATR."""
    df = _frame(10)
    df["close"] = [100, 100, 100, 96, 92, 92, 92, 92, 92, 92]
    df["atr_14"] = 2.0
    triggered = compute_atr_stop(df, entry_close=100.0, atr_multiplier=2.0)
    # Stop level = 100 - 2*2 = 96 → triggered when close < 96
    assert triggered.iloc[3] == False  # 96 not < 96  # noqa: E712
    assert triggered.iloc[4] == True  # 92 < 96  # noqa: E712


def test_validate_columns_raises():
    """Missing required columns raises ValueError."""
    bad = _frame(5).drop(columns=["hurst_dfa"])
    with pytest.raises(ValueError, match="missing required columns"):
        generate_entry_signals(bad)
