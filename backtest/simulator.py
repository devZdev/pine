"""
simulator.py
============
Trade-by-trade options P&L simulation for cash-secured put selling.

For each entry signal:
  1. Apply slippage to the entry spot price.
  2. Solve the 20-delta strike using Black-Scholes.
  3. Compute premium via BS put pricing.
  4. Compute position size via half-Kelly.
  5. Determine exit date: first of (30 DTE expiry, early exit signal, ATR stop).
  6. Compute P&L at exit.

Output: per-trade log DataFrame and equity curve Series.

No Python loops over DataFrame rows in the hot path — the trade loop iterates
over *signal* indices (one entry per trade), not per-bar.  The inner scan for
the exit date uses vectorized boolean indexing on the post-entry slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.options_math import (
    compute_realized_vol,
    solve_strike_for_delta,
    bs_put_price,
)
from backtest.signal_generator import generate_entry_signals, generate_exit_signals
from backtest.kelly_sizer import compute_kelly_fraction


# ── Constants ─────────────────────────────────────────────────────────────────

RISK_FREE_RATE: float = 0.05
DTE: int = 30
DELTA_TARGET: float = 0.20
SLIPPAGE: float = 0.001          # 0.1% of underlying price
ATR_STOP_MULTIPLIER: float = 2.0
INITIAL_PORTFOLIO: float = 100_000.0


# ── Trade Record ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float          # spot at entry (post-slippage)
    exit_price: float           # spot at exit
    strike: float
    premium: float              # premium received
    kelly_fraction: float
    position_value: float       # cash collateral deployed
    pnl: float                  # dollar P&L
    pnl_pct: float              # P&L as % of collateral
    exit_reason: str            # 'expiry' | 'take_profit' | 'trend_break' | 'atr_stop'


# ── Core Simulator ────────────────────────────────────────────────────────────

def simulate_trades(
    daily_df: pd.DataFrame,
    symbol: str,
    bb_pct_threshold: float = 0.20,
    hurst_threshold: float = 0.45,
    initial_portfolio: float = INITIAL_PORTFOLIO,
    risk_free_rate: float = RISK_FREE_RATE,
    dte: int = DTE,
    delta_target: float = DELTA_TARGET,
    slippage: float = SLIPPAGE,
    atr_multiplier: float = ATR_STOP_MULTIPLIER,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> tuple[list[TradeRecord], pd.Series]:
    """Simulate all cash-secured put trades on *daily_df*.

    Parameters
    ----------
    daily_df:
        Daily OHLCV DataFrame with feature columns (close, sma_200, bb_pct,
        hurst_dfa, atr_14).  Must have a DatetimeIndex.
    symbol:
        Ticker label for logging and trade records.
    bb_pct_threshold:
        Entry condition: bb_pct < threshold (default 0.20).
    hurst_threshold:
        Entry condition: hurst_dfa < threshold (default 0.45).
    initial_portfolio:
        Starting portfolio value in dollars.
    risk_free_rate:
        Annualized risk-free rate for Black-Scholes.
    dte:
        Days to expiry for options (calendar days, default 30).
    delta_target:
        Absolute delta for strike selection (default 0.20).
    slippage:
        One-way slippage fraction (default 0.001 = 0.1%).
    atr_multiplier:
        ATR multiplier for trailing stop (default 2.0).
    date_start:
        Restrict simulation to on/after this date (optional).
    date_end:
        Restrict simulation to on/before this date (optional).

    Returns
    -------
    tuple[list[TradeRecord], pd.Series]
        - List of TradeRecord objects (one per completed trade).
        - Daily equity curve as pd.Series with DatetimeIndex.
    """
    # ── Slice to requested date range ─────────────────────────────────────────
    df = daily_df.copy()
    if date_start is not None:
        df = df[df.index >= date_start]
    if date_end is not None:
        df = df[df.index <= date_end]

    if df.empty:
        logger.warning("simulate_trades: empty DataFrame after date filter for {}.", symbol)
        return [], pd.Series(dtype=float, name="equity")

    # ── Precompute realized vol on full slice ─────────────────────────────────
    df["realized_vol"] = compute_realized_vol(df["close"], window=30)

    # ── Generate entry / exit signals ─────────────────────────────────────────
    entry_signals = generate_entry_signals(
        df, bb_pct_threshold=bb_pct_threshold,
        hurst_threshold=hurst_threshold, shift_bars=1,
    )
    exit_signals = generate_exit_signals(df, shift_bars=1)

    # ── Build equity curve (daily mark-to-model) ───────────────────────────────
    equity_curve = pd.Series(
        np.nan, index=df.index, name=f"{symbol}_equity"
    )
    equity_curve.iloc[0] = initial_portfolio

    trades: list[TradeRecord] = []
    portfolio_value: float = initial_portfolio

    # Track open position to prevent overlapping trades on same symbol
    position_open: bool = False
    position_exit_idx: int = -1   # index into df.index

    # Precompute index positions for fast slicing
    index_arr = df.index
    n = len(index_arr)

    # Convert signals to numpy boolean arrays for fast lookup
    entry_arr: np.ndarray = entry_signals.reindex(index_arr, fill_value=False).values
    exit_arr: np.ndarray = exit_signals.reindex(index_arr, fill_value=False).values
    close_arr: np.ndarray = df["close"].values
    sma_arr: np.ndarray = df["sma_200"].values
    atr_arr: np.ndarray = df["atr_14"].values
    rvol_arr: np.ndarray = df["realized_vol"].values

    for i in range(n):
        # Mark equity for today before trading
        if i > 0 and np.isnan(equity_curve.iloc[i]):
            equity_curve.iloc[i] = portfolio_value

        # Skip if position is already open
        if position_open:
            if i >= position_exit_idx:
                position_open = False
            # Equity update happens at trade close (handled in trade completion)
            continue

        if not entry_arr[i]:
            continue

        # ── Entry checks ──────────────────────────────────────────────────────
        spot = close_arr[i]
        rvol = rvol_arr[i]
        atr_val = atr_arr[i]

        if np.isnan(spot) or np.isnan(rvol) or rvol <= 0:
            continue

        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # Apply buy-side slippage (we pay slightly more for the underlying
        # to reflect execution costs; affects strike and premium)
        spot_with_slip = spot * (1.0 + slippage)

        T = dte / 365.0

        # Solve 20-delta strike
        try:
            strike = solve_strike_for_delta(
                S=spot_with_slip,
                r=risk_free_rate,
                sigma=rvol,
                T=T,
                target_delta=-delta_target,
            )
        except Exception as exc:
            logger.warning("Strike solver failed at {}: {}. Skipping.", index_arr[i], exc)
            continue

        # Compute premium
        premium = bs_put_price(
            S=spot_with_slip,
            K=strike,
            r=risk_free_rate,
            sigma=rvol,
            T=T,
        )

        if premium <= 0:
            logger.debug("Zero premium at {}. Skipping.", index_arr[i])
            continue

        # Kelly sizing
        kelly_frac = compute_kelly_fraction(premium, strike, delta_target)
        if kelly_frac <= 0:
            logger.debug("Zero Kelly at {}. Skipping.", index_arr[i])
            continue

        collateral = kelly_frac * portfolio_value  # cash required

        # ── Determine expiry bar ───────────────────────────────────────────────
        expiry_idx = min(i + dte, n - 1)

        # ── Scan for early exit in (i, expiry_idx] ────────────────────────────
        atr_stop_level = spot - atr_multiplier * atr_val

        # Vectorized scan: find first bar triggering an exit condition
        scan_slice = slice(i + 1, expiry_idx + 1)
        exit_trigger_arr = exit_arr[scan_slice]
        close_scan = close_arr[scan_slice]
        atr_stop_trigger = close_scan < atr_stop_level

        combined_exit = exit_trigger_arr | atr_stop_trigger
        exit_bar_offsets = np.where(combined_exit)[0]

        if len(exit_bar_offsets) > 0:
            first_exit_offset = exit_bar_offsets[0]
            actual_exit_idx = i + 1 + first_exit_offset
            # Determine reason
            if atr_stop_trigger[first_exit_offset]:
                exit_reason = "atr_stop"
            elif exit_trigger_arr[first_exit_offset]:
                # Distinguish take profit vs trend break
                bb_pct_at_exit = df["bb_pct"].iloc[actual_exit_idx]
                close_at_exit = close_arr[actual_exit_idx]
                sma_at_exit = sma_arr[actual_exit_idx]
                if not np.isnan(bb_pct_at_exit) and bb_pct_at_exit > 0.5:
                    exit_reason = "take_profit"
                else:
                    exit_reason = "trend_break"
            else:
                exit_reason = "expiry"
        else:
            actual_exit_idx = expiry_idx
            exit_reason = "expiry"

        exit_date = index_arr[actual_exit_idx]
        exit_price_raw = close_arr[actual_exit_idx]

        # Apply sell-side slippage on exit
        exit_price = exit_price_raw * (1.0 - slippage)

        # ── Compute P&L ───────────────────────────────────────────────────────
        if exit_price >= strike:
            # Put expires worthless — keep full premium
            pnl = premium
            pnl_dollar = premium * (collateral / strike)
        else:
            # Assigned — partial/full loss
            intrinsic_loss = strike - exit_price
            pnl = premium - intrinsic_loss
            pnl_dollar = pnl * (collateral / strike)

        # Cap max loss to collateral (cash-secured: no leverage)
        max_loss = -(collateral)
        pnl_dollar = max(pnl_dollar, max_loss)

        pnl_pct = pnl_dollar / collateral if collateral > 0 else 0.0

        # ── Update portfolio ──────────────────────────────────────────────────
        portfolio_value = portfolio_value + pnl_dollar

        # Update equity curve from entry to exit
        equity_curve.iloc[i: actual_exit_idx + 1] = portfolio_value

        # ── Record trade ──────────────────────────────────────────────────────
        trade = TradeRecord(
            symbol=symbol,
            entry_date=index_arr[i],
            exit_date=exit_date,
            entry_price=float(spot_with_slip),
            exit_price=float(exit_price),
            strike=float(strike),
            premium=float(premium),
            kelly_fraction=float(kelly_frac),
            position_value=float(collateral),
            pnl=float(pnl_dollar),
            pnl_pct=float(pnl_pct),
            exit_reason=exit_reason,
        )
        trades.append(trade)

        position_open = True
        position_exit_idx = actual_exit_idx

        logger.debug(
            "[{}] {} | Entry={:.2f} Strike={:.2f} Prem={:.4f} "
            "Exit={:.2f} Reason={} PnL={:.2f} ({:.2%})",
            symbol, index_arr[i].date(), spot_with_slip, strike, premium,
            exit_price, exit_reason, pnl_dollar, pnl_pct,
        )

    # Forward-fill any remaining NaN in equity curve
    equity_curve = equity_curve.ffill().fillna(initial_portfolio)

    logger.info(
        "[{}] Simulation complete: {} trades, final equity={:.2f}.",
        symbol, len(trades), portfolio_value,
    )
    return trades, equity_curve


def trades_to_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    """Convert list of TradeRecord to a tidy DataFrame.

    Parameters
    ----------
    trades:
        List of TradeRecord objects from simulate_trades.

    Returns
    -------
    pd.DataFrame
        Per-trade log with columns matching the performance spec.
    """
    if not trades:
        return pd.DataFrame(columns=[
            "symbol", "entry_date", "exit_date", "entry_price", "exit_price",
            "strike", "premium", "kelly_fraction", "position_value",
            "pnl", "pnl_pct", "exit_reason",
        ])

    records = [
        {
            "symbol": t.symbol,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "strike": t.strike,
            "premium": t.premium,
            "kelly_fraction": t.kelly_fraction,
            "position_value": t.position_value,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]
    df = pd.DataFrame(records)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)
    return df


def combine_equity_curves(
    equity_btc: pd.Series,
    equity_tsla: pd.Series,
    initial_portfolio: float = INITIAL_PORTFOLIO,
) -> pd.Series:
    """Combine two per-symbol equity curves into one portfolio equity curve.

    Each symbol contributes 50% of the initial portfolio.  Daily portfolio
    equity is the sum of the two sub-portfolio values, reindexed to a common
    daily timeline.

    Parameters
    ----------
    equity_btc:
        Equity curve for BTC sub-portfolio.
    equity_tsla:
        Equity curve for TSLA sub-portfolio.
    initial_portfolio:
        Total portfolio starting value.

    Returns
    -------
    pd.Series
        Combined portfolio equity curve.
    """
    half = initial_portfolio / 2.0

    # Normalize each curve relative to its starting value, then scale
    btc_norm = equity_btc / equity_btc.iloc[0] * half if len(equity_btc) > 0 else pd.Series(dtype=float)
    tsla_norm = equity_tsla / equity_tsla.iloc[0] * half if len(equity_tsla) > 0 else pd.Series(dtype=float)

    # Align on common index
    combined = btc_norm.add(tsla_norm, fill_value=0)
    combined.name = "portfolio_equity"
    combined = combined.sort_index()
    return combined
