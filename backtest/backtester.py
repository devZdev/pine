"""
backtester.py
=============
Orchestrator: loads data, resamples 5m → daily, runs IS/OOS backtests,
optional WFO, builds benchmark, assembles and saves performance matrix.

Uses vectorbt for portfolio stats cross-validation where available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.simulator import (
    simulate_trades,
    trades_to_dataframe,
    combine_equity_curves,
    INITIAL_PORTFOLIO,
)
from backtest.performance import (
    compute_metrics,
    compute_benchmark_metrics,
    build_buyhold_benchmark,
    metrics_to_dict,
    metrics_to_raw_dict,
    PerformanceMetrics,
)
from backtest.wfo_engine import run_all_folds, ParamSet

try:
    from tabulate import tabulate
    _TABULATE_AVAILABLE = True
except ImportError:
    _TABULATE_AVAILABLE = False
    logger.warning("tabulate not installed — plain text table output will be used.")

try:
    import vectorbt as vbt
    _VBT_AVAILABLE = True
    logger.info("vectorbt {} available for cross-validation.", vbt.__version__)
except ImportError:
    _VBT_AVAILABLE = False
    logger.warning("vectorbt not installed — VBT cross-validation skipped.")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file and ensure DatetimeIndex with UTC timezone."""
    logger.info("Loading parquet: {}", path)
    df = pd.read_parquet(path)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Localize to UTC if tz-naive
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    logger.info("  Loaded {} rows ({} – {}).", len(df), df.index[0].date(), df.index[-1].date())
    return df


def resample_to_daily(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute OHLCV + features to daily bars.

    OHLCV: standard open/high/low/close/volume resampling.
    Feature columns: last value of the day (represents end-of-day state,
    consistent with no look-ahead since features are already lagged).

    Parameters
    ----------
    df_5m:
        5-minute OHLCV + feature DataFrame with DatetimeIndex (UTC).

    Returns
    -------
    pd.DataFrame
        Daily OHLCV + features, DatetimeIndex (UTC), no intraday data.
    """
    logger.info("Resampling 5m → daily ({} 5m bars).", len(df_5m))

    ohlcv_cols = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Use only columns that exist
    agg_dict = {k: v for k, v in ohlcv_cols.items() if k in df_5m.columns}

    # Feature columns — take last value of each day
    feature_cols = ["atr_14", "sma_200", "bb_upper", "bb_lower", "bb_pct", "hurst_dfa"]
    for col in feature_cols:
        if col in df_5m.columns:
            agg_dict[col] = "last"

    if "is_filled" in df_5m.columns:
        agg_dict["is_filled"] = "any"

    daily = df_5m.resample("1D").agg(agg_dict)

    # Drop days with no data (weekends, holidays)
    daily = daily.dropna(subset=["close"])

    logger.info("  Daily bars: {} ({} – {}).", len(daily), daily.index[0].date(), daily.index[-1].date())
    return daily


# ── VBT Cross-Validation ──────────────────────────────────────────────────────

def vbt_cross_validate(
    equity: pd.Series,
    trade_log: pd.DataFrame,
    label: str,
    risk_free_rate: float = 0.05,
) -> None:
    """Use vectorbt to cross-validate key metrics vs manual calculation.

    This logs the VBT stats alongside our manual calculations for sanity
    checking.  Does not overwrite the manual metrics.
    """
    if not _VBT_AVAILABLE:
        return

    try:
        # Build a minimal vbt portfolio from the equity curve
        # Use from_holding to wrap the equity curve
        daily_returns = equity.pct_change().dropna()

        if len(daily_returns) < 5:
            return

        # Compute Sharpe and Sortino via VBT's ReturnsAccessor
        returns_accessor = vbt.returns.accessors.ReturnsAccessor(daily_returns)
        vbt_sharpe = returns_accessor.sharpe_ratio(
            levy_alpha=2.0,
            required_return=risk_free_rate / 252,
            annualization=252,
        )
        vbt_sortino = returns_accessor.sortino_ratio(
            required_return=risk_free_rate / 252,
            annualization=252,
        )
        logger.info(
            "[{}] VBT cross-val | Sharpe={:.4f} Sortino={:.4f}",
            label, vbt_sharpe, vbt_sortino,
        )
    except Exception as exc:
        logger.debug("VBT cross-validation skipped for '{}': {}", label, exc)


# ── Portfolio Assembly ─────────────────────────────────────────────────────────

def run_period(
    btc_daily: pd.DataFrame,
    tsla_daily: pd.DataFrame,
    params: ParamSet,
    date_start: str,
    date_end: str,
    label: str,
    initial_portfolio: float = INITIAL_PORTFOLIO,
) -> tuple[PerformanceMetrics, pd.Series, pd.DataFrame]:
    """Run the full combined portfolio simulation for one time period.

    Parameters
    ----------
    btc_daily, tsla_daily:
        Daily DataFrames with features.
    params:
        Strategy parameters to use.
    date_start, date_end:
        Period boundaries.
    label:
        Metrics label (e.g. 'IS', 'OOS').
    initial_portfolio:
        Starting equity.

    Returns
    -------
    tuple[PerformanceMetrics, pd.Series, pd.DataFrame]
        (metrics, combined_equity_curve, trade_log)
    """
    start_ts = pd.Timestamp(date_start)
    end_ts = pd.Timestamp(date_end)

    btc_trades, btc_equity = simulate_trades(
        daily_df=btc_daily,
        symbol="BTC",
        bb_pct_threshold=params.bb_pct_threshold,
        hurst_threshold=params.hurst_threshold,
        initial_portfolio=initial_portfolio / 2.0,
        date_start=start_ts,
        date_end=end_ts,
    )

    tsla_trades, tsla_equity = simulate_trades(
        daily_df=tsla_daily,
        symbol="TSLA",
        bb_pct_threshold=params.bb_pct_threshold,
        hurst_threshold=params.hurst_threshold,
        initial_portfolio=initial_portfolio / 2.0,
        date_start=start_ts,
        date_end=end_ts,
    )

    combined_equity = combine_equity_curves(btc_equity, tsla_equity, initial_portfolio)
    all_trades = btc_trades + tsla_trades
    trade_df = trades_to_dataframe(all_trades)

    metrics = compute_metrics(combined_equity, trade_df, label=label)
    vbt_cross_validate(combined_equity, trade_df, label)

    return metrics, combined_equity, trade_df


# ── Output Formatting ─────────────────────────────────────────────────────────

def print_performance_table(metrics_list: list[PerformanceMetrics]) -> None:
    """Print a formatted performance matrix to stdout."""
    rows = [metrics_to_dict(m) for m in metrics_list]
    df = pd.DataFrame(rows)

    if _TABULATE_AVAILABLE:
        table_str = tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False)
        logger.info("\n{}", table_str)
    else:
        logger.info("\n{}", df.to_string(index=False))


def save_performance_csv(
    metrics_list: list[PerformanceMetrics],
    output_path: Path,
) -> None:
    """Save performance matrix to CSV."""
    rows = [metrics_to_raw_dict(m) for m in metrics_list]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success("Performance matrix saved: {}", output_path)


def save_trade_log(
    trade_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save per-trade log to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(output_path, index=False)
    logger.success("Trade log saved: {} ({} trades).", output_path, len(trade_df))


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def run_backtest(
    data_dir: Path,
    output_dir: Path,
    is_end: str = "2023-12-31",
    oos_start: str = "2024-01-01",
    oos_end: str = "2026-05-06",
    run_wfo: bool = True,
    default_params: Optional[ParamSet] = None,
    initial_portfolio: float = INITIAL_PORTFOLIO,
) -> dict:
    """Full backtest orchestration.

    Parameters
    ----------
    data_dir:
        Path to directory containing parquet files.
    output_dir:
        Path to output directory for results.
    is_end:
        Last date of in-sample period.
    oos_start:
        First date of out-of-sample period.
    oos_end:
        Last date of backtest.
    run_wfo:
        Whether to run walk-forward optimization (default True).
    default_params:
        Parameter set to use if run_wfo=False (defaults to bb=0.20, hurst=0.45).
    initial_portfolio:
        Starting capital.

    Returns
    -------
    dict
        Keys: 'metrics_list', 'trade_log', 'equity_curves', 'wfo_results'.
    """
    logger.info("=" * 70)
    logger.info("Glass Box Options Backtester — Starting")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    btc_5m_path = data_dir / "BTC_USD_5m.parquet"
    tsla_5m_path = data_dir / "TSLA_5m.parquet"

    if not btc_5m_path.exists():
        raise FileNotFoundError(f"BTC 5m parquet not found: {btc_5m_path}")
    if not tsla_5m_path.exists():
        raise FileNotFoundError(f"TSLA 5m parquet not found: {tsla_5m_path}")

    btc_5m = load_parquet(btc_5m_path)
    tsla_5m = load_parquet(tsla_5m_path)

    # ── Resample to daily ─────────────────────────────────────────────────────
    btc_daily = resample_to_daily(btc_5m)
    tsla_daily = resample_to_daily(tsla_5m)

    # ── Default parameters ────────────────────────────────────────────────────
    if default_params is None:
        default_params = ParamSet(bb_pct_threshold=0.20, hurst_threshold=0.45)

    # ── WFO ───────────────────────────────────────────────────────────────────
    wfo_results = []
    wfo_fold_metrics: list[PerformanceMetrics] = []
    final_params = default_params

    if run_wfo:
        logger.info("Running Walk-Forward Optimization...")
        wfo_results, final_params = run_all_folds(
            btc_daily, tsla_daily, initial_portfolio
        )
        for fr in wfo_results:
            wfo_fold_metrics.append(fr.test_metrics)
        logger.success("WFO complete. Final OOS params: {}", final_params)
    else:
        logger.info("WFO disabled — using default params: {}", default_params)
        final_params = default_params

    # ── IS Backtest (2020-01-01 → is_end) ────────────────────────────────────
    is_start = "2020-01-01"
    logger.info("Running IS backtest: {} – {}", is_start, is_end)
    is_metrics, is_equity, is_trades = run_period(
        btc_daily, tsla_daily, final_params,
        is_start, is_end, "IS",
        initial_portfolio,
    )

    # ── OOS Backtest (oos_start → oos_end) ────────────────────────────────────
    logger.info("Running OOS backtest: {} – {}", oos_start, oos_end)
    oos_metrics, oos_equity, oos_trades = run_period(
        btc_daily, tsla_daily, final_params,
        oos_start, oos_end, "OOS",
        initial_portfolio,
    )

    # ── Combined (IS + OOS) ───────────────────────────────────────────────────
    logger.info("Running Combined backtest: {} – {}", is_start, oos_end)
    combined_metrics, combined_equity, combined_trades = run_period(
        btc_daily, tsla_daily, final_params,
        is_start, oos_end, "Combined",
        initial_portfolio,
    )

    # ── Buy & Hold Benchmark ──────────────────────────────────────────────────
    logger.info("Building Buy & Hold benchmark...")
    bh_metrics = compute_benchmark_metrics(
        btc_daily, tsla_daily,
        start_date=is_start, end_date=oos_end,
        initial_portfolio=initial_portfolio,
    )
    bh_equity = build_buyhold_benchmark(
        btc_daily, tsla_daily,
        start_date=is_start, end_date=oos_end,
        initial_portfolio=initial_portfolio,
    )

    # ── Assemble results ──────────────────────────────────────────────────────
    all_metrics: list[PerformanceMetrics] = (
        [is_metrics, oos_metrics, combined_metrics, bh_metrics]
        + wfo_fold_metrics
    )

    all_trades_df = pd.concat(
        [is_trades, oos_trades],
        ignore_index=True,
    ).drop_duplicates(subset=["symbol", "entry_date"]).sort_values("entry_date")

    # ── Print table ───────────────────────────────────────────────────────────
    print_performance_table(all_metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_performance_csv(all_metrics, output_dir / "performance_matrix.csv")
    save_trade_log(all_trades_df, output_dir / "trade_log.csv")

    # Save equity curves
    equity_df = pd.DataFrame({
        "IS_equity": is_equity,
        "OOS_equity": oos_equity,
        "Combined_equity": combined_equity,
        "BenchmarkBH_equity": bh_equity,
    })
    equity_path = output_dir / "equity_curves.csv"
    equity_df.to_csv(equity_path)
    logger.success("Equity curves saved: {}", equity_path)

    logger.success("Backtest complete. Results in: {}", output_dir)

    return {
        "metrics_list": all_metrics,
        "trade_log": all_trades_df,
        "equity_curves": equity_df,
        "wfo_results": wfo_results,
        "final_params": final_params,
        "is_equity": is_equity,
        "oos_equity": oos_equity,
        "benchmark_equity": bh_equity,
    }
