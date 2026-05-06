"""
main_backtest.py
================
CLI entry point for the Glass Box Options Backtester.

Usage
-----
    python main_backtest.py [OPTIONS]

Options
-------
    --data-dir PATH       Path to directory containing parquet files
                          (default: data/raw)
    --output-dir PATH     Path to output directory for results
                          (default: backtest/results)
    --is-end DATE         Last date of in-sample period (default: 2023-12-31)
    --oos-start DATE      First date of out-of-sample period (default: 2024-01-01)
    --oos-end DATE        Last date of out-of-sample period (default: 2026-05-06)
    --no-wfo              Disable walk-forward optimization
    --bb-pct FLOAT        bb_pct entry threshold (used when --no-wfo)
    --hurst FLOAT         hurst_dfa entry threshold (used when --no-wfo)
    --initial-capital FLOAT  Starting portfolio value (default: 100000)
    --log-level LEVEL     Loguru log level (DEBUG/INFO/SUCCESS/WARNING/ERROR)
    --help                Show this message and exit

Examples
--------
    # Full backtest with WFO (default)
    python main_backtest.py

    # Custom data dir, skip WFO
    python main_backtest.py --data-dir /path/to/parquets --no-wfo

    # Custom parameters without WFO
    python main_backtest.py --no-wfo --bb-pct 0.15 --hurst 0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from backtest.backtester import run_backtest
from backtest.wfo_engine import ParamSet


# ── Logging Setup ─────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """Configure loguru with a clean format and specified level."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    # Also write to a rotating log file in the output directory
    logger.add(
        "backtest/results/backtest.log",
        level="DEBUG",
        rotation="10 MB",
        retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} — {message}",
    )


# ── Argument Parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="main_backtest.py",
        description="Glass Box Options Backtester — Cash-Secured Put Selling on BTC & TSLA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to directory containing parquet files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest/results"),
        help="Path to output directory for results (default: backtest/results)",
    )
    parser.add_argument(
        "--is-end",
        type=str,
        default="2023-12-31",
        help="Last date of in-sample period (default: 2023-12-31)",
    )
    parser.add_argument(
        "--oos-start",
        type=str,
        default="2024-01-01",
        help="First date of out-of-sample period (default: 2024-01-01)",
    )
    parser.add_argument(
        "--oos-end",
        type=str,
        default="2026-05-06",
        help="Last date of out-of-sample period (default: 2026-05-06)",
    )
    parser.add_argument(
        "--no-wfo",
        action="store_true",
        default=False,
        help="Disable walk-forward optimization (use default or --bb-pct/--hurst params)",
    )
    parser.add_argument(
        "--bb-pct",
        type=float,
        default=0.20,
        help="bb_pct entry threshold for --no-wfo mode (default: 0.20)",
    )
    parser.add_argument(
        "--hurst",
        type=float,
        default=0.45,
        help="hurst_dfa entry threshold for --no-wfo mode (default: 0.45)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Starting portfolio value in dollars (default: 100000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"],
        help="Loguru log level (default: INFO)",
    )

    return parser


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> int:
    """Main CLI entry point.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args()

    # Setup output dir early so log file can be written there
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.log_level)

    logger.info("Glass Box Options Backtester — Starting")
    logger.info("  Data dir:        {}", args.data_dir)
    logger.info("  Output dir:      {}", args.output_dir)
    logger.info("  IS end:          {}", args.is_end)
    logger.info("  OOS start:       {}", args.oos_start)
    logger.info("  OOS end:         {}", args.oos_end)
    logger.info("  WFO enabled:     {}", not args.no_wfo)
    logger.info("  Initial capital: ${:,.0f}", args.initial_capital)

    # Validate date ordering
    from datetime import date
    try:
        is_end_dt = date.fromisoformat(args.is_end)
        oos_start_dt = date.fromisoformat(args.oos_start)
        oos_end_dt = date.fromisoformat(args.oos_end)
    except ValueError as exc:
        logger.error("Invalid date format: {}", exc)
        return 1

    if oos_start_dt <= is_end_dt:
        logger.error(
            "--oos-start ({}) must be after --is-end ({}).",
            args.oos_start, args.is_end,
        )
        return 1

    if oos_end_dt <= oos_start_dt:
        logger.error(
            "--oos-end ({}) must be after --oos-start ({}).",
            args.oos_end, args.oos_start,
        )
        return 1

    # Build default params for no-WFO mode
    default_params = ParamSet(
        bb_pct_threshold=args.bb_pct,
        hurst_threshold=args.hurst,
    )

    try:
        results = run_backtest(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            is_end=args.is_end,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            run_wfo=not args.no_wfo,
            default_params=default_params,
            initial_portfolio=args.initial_capital,
        )
    except FileNotFoundError as exc:
        logger.error("Data file not found: {}", exc)
        logger.error(
            "Ensure Phase 1 data pipeline has been run and parquet files "
            "exist in: {}",
            args.data_dir,
        )
        return 1
    except Exception as exc:
        logger.exception("Backtest failed with unexpected error: {}", exc)
        return 1

    # Final summary
    metrics_list = results["metrics_list"]
    is_m = next((m for m in metrics_list if m.label == "IS"), None)
    oos_m = next((m for m in metrics_list if m.label == "OOS"), None)
    bh_m = next((m for m in metrics_list if m.label == "BuyHold_Benchmark"), None)

    logger.success("=" * 60)
    logger.success("BACKTEST COMPLETE")
    logger.success("=" * 60)
    if is_m:
        logger.success(
            "IS  ({} – {}): Return={:.2%}  Calmar={:.3f}  MDD={:.2%}",
            is_m.start_date, is_m.end_date,
            is_m.total_return, is_m.calmar_ratio, is_m.max_drawdown,
        )
    if oos_m:
        logger.success(
            "OOS ({} – {}): Return={:.2%}  Calmar={:.3f}  MDD={:.2%}",
            oos_m.start_date, oos_m.end_date,
            oos_m.total_return, oos_m.calmar_ratio, oos_m.max_drawdown,
        )
    if bh_m:
        logger.success(
            "B&H ({} – {}): Return={:.2%}  Calmar={:.3f}  MDD={:.2%}",
            bh_m.start_date, bh_m.end_date,
            bh_m.total_return, bh_m.calmar_ratio, bh_m.max_drawdown,
        )
    logger.success("Results saved to: {}", args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
