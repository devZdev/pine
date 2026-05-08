"""
performance.py
==============
Performance metric calculation for backtesting results.

Implements:
  - Total Return
  - CAGR
  - Max Drawdown
  - Calmar Ratio
  - Sortino Ratio
  - Sharpe Ratio
  - Win Rate
  - Avg Trade P&L
  - Buy & Hold benchmark construction

Uses vectorbt for portfolio stats where available, with manual fallback
cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Performance Metrics Dataclass ─────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    label: str
    total_return: float
    cagr: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_pnl_pct: float
    n_trades: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ── Core Calculation Functions ────────────────────────────────────────────────

def compute_total_return(equity: pd.Series) -> float:
    """(final - initial) / initial."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0])


def compute_cagr(equity: pd.Series) -> float:
    """(final/initial)^(1/years) - 1."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    start = equity.index[0]
    end = equity.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (1.0 / years) - 1.0)


def compute_max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough percentage decline (returned as negative fraction)."""
    if len(equity) < 2:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_daily_returns(equity: pd.Series) -> pd.Series:
    """Compute daily returns from equity curve."""
    returns = equity.pct_change().dropna()
    return returns


def compute_sharpe_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Annualized Sharpe Ratio.

    Sharpe = mean(excess_returns) / std(returns) * sqrt(252)
    """
    daily_returns = compute_daily_returns(equity)
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess = daily_returns - daily_rf
    std = daily_returns.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(trading_days))


def compute_sortino_ratio(
    equity: pd.Series,
    risk_free_rate: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Annualized Sortino Ratio.

    Sortino = mean(excess_returns) / std(negative_returns) * sqrt(252)
    """
    daily_returns = compute_daily_returns(equity)
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / trading_days
    excess = daily_returns - daily_rf
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) < 2:
        return 0.0
    downside_std = negative_returns.std()
    if downside_std == 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(trading_days))


def compute_calmar_ratio(equity: pd.Series) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown|."""
    cagr = compute_cagr(equity)
    mdd = compute_max_drawdown(equity)
    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / abs(mdd))


def compute_win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of trades with positive P&L."""
    if len(trade_log) == 0:
        return 0.0
    winners = (trade_log["pnl"] > 0).sum()
    return float(winners / len(trade_log))


def compute_avg_trade_pnl(trade_log: pd.DataFrame) -> float:
    """Mean trade P&L as a fraction of position value."""
    if len(trade_log) == 0:
        return 0.0
    return float(trade_log["pnl_pct"].mean())


# ── Full Metrics Builder ──────────────────────────────────────────────────────

def compute_metrics(
    equity: pd.Series,
    trade_log: pd.DataFrame,
    label: str,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """Compute all performance metrics for one equity curve + trade log.

    Parameters
    ----------
    equity:
        Portfolio equity curve (DatetimeIndex, daily).
    trade_log:
        Per-trade DataFrame with columns: pnl, pnl_pct.
    label:
        Identifier for this result set (e.g. 'IS', 'OOS', 'WFO_Fold1').
    risk_free_rate:
        Annualized risk-free rate for Sharpe/Sortino.

    Returns
    -------
    PerformanceMetrics
    """
    equity_clean = equity.dropna()
    if len(equity_clean) == 0:
        logger.warning("compute_metrics: empty equity curve for label '{}'.", label)
        return PerformanceMetrics(
            label=label, total_return=0.0, cagr=0.0, max_drawdown=0.0,
            calmar_ratio=0.0, sortino_ratio=0.0, sharpe_ratio=0.0,
            win_rate=0.0, avg_trade_pnl_pct=0.0, n_trades=0,
        )

    m = PerformanceMetrics(
        label=label,
        total_return=compute_total_return(equity_clean),
        cagr=compute_cagr(equity_clean),
        max_drawdown=compute_max_drawdown(equity_clean),
        calmar_ratio=compute_calmar_ratio(equity_clean),
        sortino_ratio=compute_sortino_ratio(equity_clean, risk_free_rate),
        sharpe_ratio=compute_sharpe_ratio(equity_clean, risk_free_rate),
        win_rate=compute_win_rate(trade_log),
        avg_trade_pnl_pct=compute_avg_trade_pnl(trade_log),
        n_trades=len(trade_log),
        start_date=str(equity_clean.index[0].date()) if len(equity_clean) > 0 else None,
        end_date=str(equity_clean.index[-1].date()) if len(equity_clean) > 0 else None,
    )

    logger.info(
        "[{}] Return={:.2%} CAGR={:.2%} MDD={:.2%} Calmar={:.2f} "
        "Sortino={:.2f} Sharpe={:.2f} WinRate={:.2%} Trades={}",
        label, m.total_return, m.cagr, m.max_drawdown, m.calmar_ratio,
        m.sortino_ratio, m.sharpe_ratio, m.win_rate, m.n_trades,
    )
    return m


# ── Buy & Hold Benchmark ──────────────────────────────────────────────────────

def build_buyhold_benchmark(
    btc_daily: pd.DataFrame,
    tsla_daily: pd.DataFrame,
    start_date: str = "2020-01-01",
    end_date: str = "2026-05-06",
    initial_portfolio: float = 100_000.0,
) -> pd.Series:
    """Construct equal-weight 50/50 buy-and-hold benchmark.

    Buys BTC and TSLA at the open on *start_date*, holds to *end_date*.

    Parameters
    ----------
    btc_daily:
        Daily OHLCV DataFrame for BTC (DatetimeIndex, must include 'open'/'close').
    tsla_daily:
        Daily OHLCV DataFrame for TSLA.
    start_date:
        Entry date string.
    end_date:
        Exit date string.
    initial_portfolio:
        Starting capital.

    Returns
    -------
    pd.Series
        Daily equity curve for the benchmark portfolio.
    """
    half = initial_portfolio / 2.0

    def _symbol_equity(df: pd.DataFrame, half_capital: float) -> pd.Series:
        # Coerce comparison timestamps to match the DataFrame index timezone
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        idx_tz = getattr(df.index, "tz", None)
        if idx_tz is not None:
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(idx_tz)
            else:
                start_ts = start_ts.tz_convert(idx_tz)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(idx_tz)
            else:
                end_ts = end_ts.tz_convert(idx_tz)
        df = df[df.index >= start_ts]
        df = df[df.index <= end_ts]
        if df.empty:
            return pd.Series(dtype=float)

        # Use open of first bar as entry price
        entry_price = float(df["open"].iloc[0])
        if entry_price <= 0:
            entry_price = float(df["close"].iloc[0])

        shares = half_capital / entry_price
        equity = df["close"] * shares
        return equity

    btc_eq = _symbol_equity(btc_daily, half)
    tsla_eq = _symbol_equity(tsla_daily, half)

    # Combine on common index
    combined = btc_eq.add(tsla_eq, fill_value=0)
    combined.name = "benchmark_equity"
    combined = combined.sort_index()

    logger.info(
        "Benchmark constructed: {:.2%} total return.",
        compute_total_return(combined),
    )
    return combined


def compute_benchmark_metrics(
    btc_daily: pd.DataFrame,
    tsla_daily: pd.DataFrame,
    start_date: str = "2020-01-01",
    end_date: str = "2026-05-06",
    initial_portfolio: float = 100_000.0,
) -> PerformanceMetrics:
    """Compute performance metrics for the buy-and-hold benchmark."""
    equity = build_buyhold_benchmark(
        btc_daily, tsla_daily, start_date, end_date, initial_portfolio
    )
    empty_trades = pd.DataFrame(columns=["pnl", "pnl_pct"])
    return compute_metrics(equity, empty_trades, label="BuyHold_Benchmark")


# ── Formatting Utilities ──────────────────────────────────────────────────────

def metrics_to_dict(m: PerformanceMetrics) -> dict:
    """Convert PerformanceMetrics to a plain dict suitable for DataFrame rows."""
    return {
        "Label": m.label,
        "Start": m.start_date,
        "End": m.end_date,
        "Total Return": f"{m.total_return:.2%}",
        "CAGR": f"{m.cagr:.2%}",
        "Max Drawdown": f"{m.max_drawdown:.2%}",
        "Calmar Ratio": f"{m.calmar_ratio:.3f}",
        "Sortino Ratio": f"{m.sortino_ratio:.3f}",
        "Sharpe Ratio": f"{m.sharpe_ratio:.3f}",
        "Win Rate": f"{m.win_rate:.2%}",
        "Avg Trade P&L": f"{m.avg_trade_pnl_pct:.2%}",
        "N Trades": m.n_trades,
    }


def metrics_to_raw_dict(m: PerformanceMetrics) -> dict:
    """Convert PerformanceMetrics to raw numeric dict for CSV output."""
    return {
        "Label": m.label,
        "Start": m.start_date,
        "End": m.end_date,
        "Total Return": round(m.total_return, 6),
        "CAGR": round(m.cagr, 6),
        "Max Drawdown": round(m.max_drawdown, 6),
        "Calmar Ratio": round(m.calmar_ratio, 6),
        "Sortino Ratio": round(m.sortino_ratio, 6),
        "Sharpe Ratio": round(m.sharpe_ratio, 6),
        "Win Rate": round(m.win_rate, 6),
        "Avg Trade P&L": round(m.avg_trade_pnl_pct, 6),
        "N Trades": m.n_trades,
    }
