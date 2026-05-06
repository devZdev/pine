"""
signal_generator.py
===================
Glass Box entry and exit signal generation — fully vectorized.

Entry signal (ALL conditions must be true simultaneously):
  1. bb_pct < bb_pct_threshold   — price near lower Bollinger band
  2. close > sma_200             — above long-term trend
  3. hurst_dfa < hurst_threshold — mean-reverting regime

Exit signal (EITHER condition triggers close):
  1. bb_pct > 0.5                — price recovered to mid-band (take profit)
  2. close < sma_200             — trend breakdown (cut loss)

Stop loss:
  ATR trailing stop: close < entry_close - 2 × atr_14

Latency simulation:
  All signals are shifted forward by 1 bar to model 500ms execution delay.
  This is applied via signal.shift(1) so no bar can trade on its own signal.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


def generate_entry_signals(
    df: pd.DataFrame,
    bb_pct_threshold: float = 0.20,
    hurst_threshold: float = 0.45,
    shift_bars: int = 1,
) -> pd.Series:
    """Compute vectorized entry signals for the Glass Box strategy.

    A boolean Series is True on bars where all three entry conditions are met.
    The result is shifted forward by *shift_bars* to simulate execution latency
    (no look-ahead bias — the signal fires on the *next* bar after conditions
    are observed).

    Parameters
    ----------
    df:
        Daily OHLCV DataFrame with feature columns: bb_pct, sma_200,
        hurst_dfa, close.
    bb_pct_threshold:
        Upper bound for bb_pct entry condition (default 0.20).
    hurst_threshold:
        Upper bound for hurst_dfa entry condition (default 0.45).
    shift_bars:
        Number of bars to shift the signal forward (default 1).

    Returns
    -------
    pd.Series
        Boolean entry signal aligned to *df* index.  True = sell a put.
    """
    _validate_columns(df, ["bb_pct", "sma_200", "hurst_dfa", "close"])

    cond_bb: pd.Series = df["bb_pct"] < bb_pct_threshold
    cond_trend: pd.Series = df["close"] > df["sma_200"]
    cond_hurst: pd.Series = df["hurst_dfa"] < hurst_threshold

    raw_signal: pd.Series = cond_bb & cond_trend & cond_hurst

    # Drop any bars with NaN in feature columns from signal
    valid_mask: pd.Series = (
        df["bb_pct"].notna()
        & df["sma_200"].notna()
        & df["hurst_dfa"].notna()
        & df["close"].notna()
    )
    raw_signal = raw_signal & valid_mask

    # Latency shift: execute on next bar after conditions observed
    shifted: pd.Series = raw_signal.shift(shift_bars, fill_value=False)
    shifted.name = "entry_signal"

    n_signals = int(shifted.sum())
    logger.info(
        "Entry signals generated: {} signals (bb_pct<{}, hurst<{}, shifted {} bar(s)).",
        n_signals, bb_pct_threshold, hurst_threshold, shift_bars,
    )
    return shifted


def generate_exit_signals(
    df: pd.DataFrame,
    shift_bars: int = 1,
) -> pd.Series:
    """Compute vectorized exit signals for the Glass Box strategy.

    An exit is triggered when EITHER:
    - bb_pct > 0.5  (price recovered to mid-band, take profit), OR
    - close < sma_200 (trend breakdown, cut loss)

    The signal is shifted by *shift_bars* for latency consistency.

    Parameters
    ----------
    df:
        Daily OHLCV DataFrame with feature columns.
    shift_bars:
        Number of bars to shift the exit signal forward (default 1).

    Returns
    -------
    pd.Series
        Boolean exit signal aligned to *df* index.
    """
    _validate_columns(df, ["bb_pct", "sma_200", "close"])

    cond_tp: pd.Series = df["bb_pct"] > 0.5
    cond_breakdown: pd.Series = df["close"] < df["sma_200"]

    raw_exit: pd.Series = cond_tp | cond_breakdown

    valid_mask: pd.Series = df["bb_pct"].notna() & df["sma_200"].notna()
    raw_exit = raw_exit & valid_mask

    shifted: pd.Series = raw_exit.shift(shift_bars, fill_value=False)
    shifted.name = "exit_signal"

    logger.debug("Exit signals generated.")
    return shifted


def compute_atr_stop(
    df: pd.DataFrame,
    entry_close: float,
    atr_multiplier: float = 2.0,
) -> pd.Series:
    """Compute per-bar ATR trailing stop level for an open position.

    The stop level is fixed at entry: entry_close - atr_multiplier * atr_14
    evaluated at entry.  This function returns a boolean Series indicating
    whether the stop has been triggered on each bar.

    Parameters
    ----------
    df:
        Daily OHLCV DataFrame slice from entry onwards.
    entry_close:
        Closing price at position entry bar.
    atr_multiplier:
        ATR multiplier for the stop distance (default 2.0).

    Returns
    -------
    pd.Series
        Boolean Series: True on bars where stop is triggered.
    """
    _validate_columns(df, ["close", "atr_14"])

    # Entry ATR is the ATR value at the first row of the slice
    entry_atr = float(df["atr_14"].iloc[0])
    stop_level = entry_close - atr_multiplier * entry_atr

    triggered: pd.Series = df["close"] < stop_level
    triggered.name = "atr_stop"
    return triggered


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Assert that all required columns exist in *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"signal_generator: missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
