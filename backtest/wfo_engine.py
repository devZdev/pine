"""
wfo_engine.py
=============
Walk-Forward Optimization (WFO) engine.

Fold schedule (rolling expanding windows):
  Fold 1: Train Jan 2020 – Dec 2021 | Test Jan 2022 – Jun 2022
  Fold 2: Train Jan 2020 – Jun 2022 | Test Jul 2022 – Dec 2022
  Fold 3: Train Jan 2020 – Dec 2022 | Test Jan 2023 – Jun 2023
  Fold 4: Train Jan 2020 – Jun 2023 | Test Jul 2023 – Dec 2023

Parameter grid:
  bb_pct_threshold: [0.10, 0.15, 0.20, 0.25]
  hurst_threshold:  [0.40, 0.45, 0.50]

Optimization metric: Calmar Ratio on the test window.
The winning parameter set from the final fold (Fold 4) is applied to the full
OOS period (2024–2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import pandas as pd
from loguru import logger

from backtest.simulator import simulate_trades, trades_to_dataframe, combine_equity_curves
from backtest.performance import compute_metrics, compute_calmar_ratio, PerformanceMetrics


# ── WFO Fold Definition ───────────────────────────────────────────────────────

@dataclass
class WFOFold:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


FOLDS: list[WFOFold] = [
    WFOFold(1, "2020-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    WFOFold(2, "2020-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    WFOFold(3, "2020-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    WFOFold(4, "2020-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
]

BB_PCT_GRID: list[float] = [0.10, 0.15, 0.20, 0.25]
HURST_GRID: list[float] = [0.40, 0.45, 0.50]


# ── Parameter Set ─────────────────────────────────────────────────────────────

@dataclass
class ParamSet:
    bb_pct_threshold: float
    hurst_threshold: float

    def __str__(self) -> str:
        return f"bb={self.bb_pct_threshold:.2f}_hurst={self.hurst_threshold:.2f}"


# ── WFO Result ────────────────────────────────────────────────────────────────

@dataclass
class WFOFoldResult:
    fold: WFOFold
    best_params: ParamSet
    best_calmar_train: float
    best_calmar_test: float
    test_metrics: PerformanceMetrics
    all_param_scores: dict[str, float]   # param_str -> calmar on test


# ── Core Fold Runner ──────────────────────────────────────────────────────────

def _run_one_param_set(
    btc_daily: pd.DataFrame,
    tsla_daily: pd.DataFrame,
    params: ParamSet,
    date_start: str,
    date_end: str,
    initial_portfolio: float = 100_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a full simulation for one parameter set on a date slice.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        (combined_equity_curve, trade_log_dataframe)
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

    return combined_equity, trade_df


def run_fold(
    fold: WFOFold,
    btc_daily: pd.DataFrame,
    tsla_daily: pd.DataFrame,
    initial_portfolio: float = 100_000.0,
) -> WFOFoldResult:
    """Run one WFO fold: train on all parameter combinations, pick best by
    Calmar on test window.

    Parameters
    ----------
    fold:
        WFOFold definition with train/test date boundaries.
    btc_daily:
        Full daily BTC DataFrame with features.
    tsla_daily:
        Full daily TSLA DataFrame with features.
    initial_portfolio:
        Starting capital for simulation.

    Returns
    -------
    WFOFoldResult
    """
    logger.info(
        "WFO Fold {}: Train [{} – {}] | Test [{} – {}]",
        fold.fold_id, fold.train_start, fold.train_end,
        fold.test_start, fold.test_end,
    )

    param_grid = [
        ParamSet(bb_pct_threshold=bb, hurst_threshold=h)
        for bb, h in product(BB_PCT_GRID, HURST_GRID)
    ]

    # ── Grid search on train window ───────────────────────────────────────────
    train_scores: dict[str, tuple[float, ParamSet]] = {}  # str -> (calmar, params)

    for params in param_grid:
        try:
            equity, _ = _run_one_param_set(
                btc_daily, tsla_daily, params,
                fold.train_start, fold.train_end, initial_portfolio,
            )
            calmar = compute_calmar_ratio(equity.dropna())
            train_scores[str(params)] = (calmar, params)
            logger.debug("  Fold {} Train | {} | Calmar={:.4f}", fold.fold_id, params, calmar)
        except Exception as exc:
            logger.warning("  Fold {} Train | {} | Error: {}", fold.fold_id, params, exc)
            train_scores[str(params)] = (float("-inf"), params)

    # Pick best params by train Calmar
    best_key = max(train_scores, key=lambda k: train_scores[k][0])
    best_calmar_train, best_params = train_scores[best_key]

    logger.info(
        "Fold {} best params: {} | Train Calmar={:.4f}",
        fold.fold_id, best_params, best_calmar_train,
    )

    # ── Evaluate best params on test window ───────────────────────────────────
    test_equity, test_trades = _run_one_param_set(
        btc_daily, tsla_daily, best_params,
        fold.test_start, fold.test_end, initial_portfolio,
    )

    test_metrics = compute_metrics(
        test_equity, test_trades,
        label=f"WFO_Fold{fold.fold_id}_Test",
    )
    best_calmar_test = test_metrics.calmar_ratio

    # ── Also evaluate all params on test for reporting ────────────────────────
    all_param_test_scores: dict[str, float] = {}
    for params in param_grid:
        try:
            eq, _ = _run_one_param_set(
                btc_daily, tsla_daily, params,
                fold.test_start, fold.test_end, initial_portfolio,
            )
            all_param_test_scores[str(params)] = compute_calmar_ratio(eq.dropna())
        except Exception:
            all_param_test_scores[str(params)] = float("-inf")

    logger.success(
        "Fold {} complete | Best={} | Test Calmar={:.4f}",
        fold.fold_id, best_params, best_calmar_test,
    )

    return WFOFoldResult(
        fold=fold,
        best_params=best_params,
        best_calmar_train=best_calmar_train,
        best_calmar_test=best_calmar_test,
        test_metrics=test_metrics,
        all_param_scores=all_param_test_scores,
    )


def run_all_folds(
    btc_daily: pd.DataFrame,
    tsla_daily: pd.DataFrame,
    initial_portfolio: float = 100_000.0,
) -> tuple[list[WFOFoldResult], ParamSet]:
    """Run all 4 WFO folds and return results + final best params.

    The final best parameters are taken from the last fold (Fold 4) — the
    most recent training window — for application to the full OOS period.

    Parameters
    ----------
    btc_daily:
        Full daily BTC DataFrame with features.
    tsla_daily:
        Full daily TSLA DataFrame with features.
    initial_portfolio:
        Starting capital.

    Returns
    -------
    tuple[list[WFOFoldResult], ParamSet]
        All fold results and the final winning parameter set.
    """
    fold_results: list[WFOFoldResult] = []

    for fold in FOLDS:
        result = run_fold(fold, btc_daily, tsla_daily, initial_portfolio)
        fold_results.append(result)

    # Final params from the last fold
    final_params = fold_results[-1].best_params
    logger.success(
        "WFO complete. Final OOS parameters: {}", final_params
    )

    return fold_results, final_params
