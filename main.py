"""
main.py
=======
Orchestrates the full data ingestion and feature engineering pipeline.

Usage
-----
    python main.py [options]

Options
-------
--symbols       Space-separated list of symbols to process.
                Supported: BTC  TSLA  (default: BTC TSLA)
--timeframes    Space-separated list of timeframes.
                Supported: 1m 5m 15m 1h 1d  (default: 1m 5m)
--start         Start date in YYYY-MM-DD format  (default: 2020-01-01)
--end           End date in YYYY-MM-DD format    (default: today)
--data-dir      Root directory for parquet output  (default: data/raw)
--no-hurst      Skip the (slow) Hurst DFA computation.
--log-level     Logging level: DEBUG | INFO | WARNING  (default: INFO)

Examples
--------
    # Full run
    python main.py --symbols BTC TSLA --timeframes 1m 5m --start 2020-01-01

    # BTC only, skip Hurst for speed
    python main.py --symbols BTC --timeframes 1m --no-hurst

    # TSLA 5m quick test
    python main.py --symbols TSLA --timeframes 5m --start 2024-01-01 --end 2024-03-01
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from pipeline.coinbase_ingestor import ingest_coinbase
from pipeline.alpaca_ingestor import ingest_alpaca
from pipeline.feature_engineer import compute_all_features
from pipeline.storage import load_parquet, save_parquet
from pipeline.utils import setup_logging


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Institutional quant data ingestion + feature engineering pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "TSLA"],
        choices=["BTC", "TSLA"],
        metavar="SYMBOL",
        help="Symbols to process.  Choices: BTC TSLA.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1m", "5m"],
        metavar="TF",
        help="Candle widths to process.  E.g. 1m 5m 1h.",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Historical start date (YYYY-MM-DD, UTC).",
    )
    parser.add_argument(
        "--end",
        default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        help="Historical end date (YYYY-MM-DD, UTC).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Root directory for parquet output.",
    )
    parser.add_argument(
        "--no-hurst",
        action="store_true",
        default=False,
        help="Skip Hurst DFA feature (saves significant compute time).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return parser


# ── Graceful shutdown ─────────────────────────────────────────────────────────

_partial_results: dict[str, dict[str, pd.DataFrame]] = {
    "BTC": {},
    "TSLA": {},
}
_shutdown_requested = False


def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ANN401
    global _shutdown_requested
    logger.warning("SIGINT received — requesting graceful shutdown after current page.")
    _shutdown_requested = True


# ── BTC pipeline ──────────────────────────────────────────────────────────────

async def run_btc_pipeline(
    timeframes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: str,
    include_hurst: bool,
) -> dict[str, pd.DataFrame]:
    """Ingest BTC/USD from Coinbase and compute features for each timeframe.

    Parameters
    ----------
    timeframes:
        List of timeframe strings, e.g. ``["1m", "5m"]``.
    start, end:
        UTC date range.
    data_dir:
        Output directory.
    include_hurst:
        Whether to compute the Hurst DFA feature.

    Returns
    -------
    dict[str, pd.DataFrame]
        Timeframe → feature-enriched DataFrame.
    """
    logger.info("=== BTC pipeline starting ===")

    try:
        raw_data = await ingest_coinbase(
            timeframes=timeframes,
            start=start,
            end=end,
            base_dir=data_dir,
        )
    except KeyboardInterrupt:
        logger.warning("[BTC] Interrupted during ingestion.")
        for tf, df in _partial_results["BTC"].items():
            if df is not None and not df.empty:
                save_parquet(df, "BTC_USD", tf, data_dir)
        raise

    results: dict[str, pd.DataFrame] = {}
    for tf, df in raw_data.items():
        if df is None or df.empty:
            logger.warning("[BTC] No data for timeframe {}; skipping features.", tf)
            continue

        logger.info("[BTC] Computing features for {}.", tf)
        enriched = compute_all_features(df, include_hurst=include_hurst)
        save_parquet(enriched, "BTC_USD", tf, data_dir)
        results[tf] = enriched
        _partial_results["BTC"][tf] = enriched

    logger.info("=== BTC pipeline complete ({} timeframes). ===", len(results))
    return results


# ── TSLA pipeline ─────────────────────────────────────────────────────────────

async def run_tsla_pipeline(
    timeframes: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: str,
    include_hurst: bool,
) -> dict[str, pd.DataFrame]:
    """Ingest TSLA from Alpaca and compute features for each timeframe.

    Parameters
    ----------
    timeframes:
        List of timeframe strings, e.g. ``["1m", "5m"]``.
    start, end:
        UTC date range.
    data_dir:
        Output directory.
    include_hurst:
        Whether to compute the Hurst DFA feature.

    Returns
    -------
    dict[str, pd.DataFrame]
        Timeframe → feature-enriched DataFrame.
    """
    logger.info("=== TSLA pipeline starting ===")

    try:
        raw_data = await ingest_alpaca(
            timeframes=timeframes,
            start=start,
            end=end,
            base_dir=data_dir,
        )
    except KeyboardInterrupt:
        logger.warning("[TSLA] Interrupted during ingestion.")
        for tf, df in _partial_results["TSLA"].items():
            if df is not None and not df.empty:
                save_parquet(df, "TSLA", tf, data_dir)
        raise

    results: dict[str, pd.DataFrame] = {}
    for tf, df in raw_data.items():
        if df is None or df.empty:
            logger.warning("[TSLA] No data for timeframe {}; skipping features.", tf)
            continue

        logger.info("[TSLA] Computing features for {}.", tf)
        enriched = compute_all_features(df, include_hurst=include_hurst)
        save_parquet(enriched, "TSLA", tf, data_dir)
        results[tf] = enriched
        _partial_results["TSLA"][tf] = enriched

    logger.info("=== TSLA pipeline complete ({} timeframes). ===", len(results))
    return results


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def run_pipeline(args: argparse.Namespace) -> int:
    """Async pipeline entry point.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = error.
    """
    load_dotenv()

    setup_logging(level=args.log_level)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC").replace(
        hour=23, minute=59, second=59
    )

    logger.info(
        "Pipeline starting. symbols={} timeframes={} start={} end={} data_dir={}",
        args.symbols, args.timeframes, start.date(), end.date(), args.data_dir,
    )

    include_hurst = not args.no_hurst
    if not include_hurst:
        logger.info("Hurst DFA feature DISABLED (--no-hurst flag set).")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _handle_sigint)

    tasks: list[asyncio.Task[dict[str, pd.DataFrame]]] = []

    if "BTC" in args.symbols:
        tasks.append(
            asyncio.create_task(
                run_btc_pipeline(args.timeframes, start, end, args.data_dir, include_hurst),
                name="btc_pipeline",
            )
        )

    if "TSLA" in args.symbols:
        tasks.append(
            asyncio.create_task(
                run_tsla_pipeline(args.timeframes, start, end, args.data_dir, include_hurst),
                name="tsla_pipeline",
            )
        )

    if not tasks:
        logger.error("No valid symbols selected. Exiting.")
        return 1

    all_results: dict[str, dict[str, pd.DataFrame]] = {}

    try:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        for task, result in zip(tasks, completed):
            if isinstance(result, Exception):
                logger.error("Task '{}' failed: {}", task.get_name(), result)
            else:
                all_results[task.get_name()] = result
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted.  Cancelling remaining tasks.")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return 130  # standard exit code for SIGINT

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline complete.  Output summary:")
    for task_name, tf_map in all_results.items():
        for tf, df in tf_map.items():
            if df is not None and not df.empty:
                logger.info(
                    "  {:<20} {} bars | columns: {}",
                    f"{task_name}/{tf}", len(df), list(df.columns),
                )
    logger.info("=" * 60)

    return 0


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        logger.warning("Pipeline terminated by user.")
        exit_code = 130
    except Exception as exc:
        logger.exception("Unhandled exception in pipeline: {}", exc)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
