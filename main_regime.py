"""
main_regime.py
==============
FastAPI application entry point for the Regime API (Phase 2).

Startup sequence (lifespan)
---------------------------
1. Load environment variables from ``.env``.
2. Instantiate and warm up the in-memory DataStore (load all Parquet files).
3. Load the Chronos-T5 model onto CPU (downloads on first run, cache on
   subsequent runs).
4. Log ``Regime API ready`` once everything is up.

Shutdown sequence (lifespan exit)
----------------------------------
Loguru flushes automatically; no explicit teardown required.

Environment variables
---------------------
REGIME_HOST         Bind address (default 0.0.0.0)
REGIME_PORT         Bind port (default 8000)
DATA_DIR            Root directory for Parquet files (default data/raw)
CHRONOS_MODEL_ID    HuggingFace model ID (default amazon/chronos-t5-tiny)
HUGGINGFACE_TOKEN   HuggingFace access token (required for gated models)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

from regime.data_loader import DataStore, set_data_store
from regime.model import ChronosForecaster, set_forecaster
from regime.router import router


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Replace the default Loguru sink with a clean, levelled formatter."""
    logger.remove()  # remove default stderr sink
    logger.add(
        sys.stderr,
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager that owns startup and shutdown logic."""

    # ── 1. Environment ────────────────────────────────────────────────────────
    load_dotenv()
    _configure_logging()
    logger.info("Environment loaded.")

    # ── 2. Data store ─────────────────────────────────────────────────────────
    data_dir = os.environ.get("DATA_DIR", "data/raw")
    logger.info("Initialising DataStore from '{}'.", data_dir)
    store = DataStore(data_dir=data_dir)
    store.load_all()
    set_data_store(store)

    loaded_syms = store.loaded_symbols
    if loaded_syms:
        logger.info("Symbols in memory: {}.", loaded_syms)
    else:
        logger.warning(
            "No Parquet files found under '{}'.  "
            "Run the data pipeline (main.py) before querying /regime.",
            data_dir,
        )

    # ── 3. Chronos model ──────────────────────────────────────────────────────
    model_id = os.environ.get("CHRONOS_MODEL_ID", "amazon/chronos-t5-tiny")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    logger.info("Initialising ChronosForecaster with model '{}'.", model_id)
    forecaster = ChronosForecaster(
        model_id=model_id,
        hf_token=hf_token,
        num_samples=20,
    )

    try:
        forecaster.load()
    except ImportError as exc:
        logger.error(
            "chronos-forecasting package missing: {}.  "
            "/regime endpoint will return 503 until the model is loaded.",
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Chronos model failed to load: {}.  "
            "/regime endpoint will return 503.",
            exc,
        )

    set_forecaster(forecaster)

    # ── 4. Ready ──────────────────────────────────────────────────────────────
    logger.info("Regime API ready.")

    yield  # ── application runs here ──────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Regime API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Regime API",
    description=(
        "Zero-shot market regime classification powered by Amazon Chronos-T5. "
        "Returns TRENDING / MEAN_REVERTING labels with confidence scores and "
        "10-step ahead forecast price ranges."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Uvicorn server, reading host/port from environment."""
    import uvicorn  # type: ignore[import-untyped]

    host = os.environ.get("REGIME_HOST", "0.0.0.0")
    port = int(os.environ.get("REGIME_PORT", "8000"))

    load_dotenv()
    _configure_logging()

    logger.info("Starting Regime API on {}:{}", host, port)

    uvicorn.run(
        "main_regime:app",
        host=host,
        port=port,
        log_level="warning",   # Uvicorn's own logger — our Loguru handles app logs
        access_log=False,       # Keep stdout clean; Loguru logs requests if needed
    )


if __name__ == "__main__":
    main()
