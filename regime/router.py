"""
regime/router.py
================
FastAPI route handlers for the Regime API.

Endpoints
---------
GET  /health    — liveness check for Docker healthcheck
GET  /symbols   — list symbols with available parquet data
GET  /regime    — run Chronos inference + regime classification
POST /refresh   — reload parquet files from disk

All response schemas are Pydantic models so FastAPI generates correct
OpenAPI docs and JSON serialisation is deterministic.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from regime.classifier import classify_regime
from regime.data_loader import (
    get_data_store,
    get_latest_atr,
    get_latest_hurst,
)
from regime.model import get_forecaster


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model: str = Field(..., examples=["amazon/chronos-t5-tiny"])
    loaded: bool = Field(..., examples=[True])


class SymbolsResponse(BaseModel):
    symbols: list[str] = Field(..., examples=[["BTC", "TSLA"]])


class ForecastRange(BaseModel):
    low: float = Field(..., description="Minimum of q10 forecast band (raw price)")
    high: float = Field(..., description="Maximum of q90 forecast band (raw price)")


class RegimeResponse(BaseModel):
    symbol: str = Field(..., examples=["BTC"])
    regime: str = Field(..., examples=["MEAN_REVERTING"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.82])
    forecast_range: ForecastRange
    hurst: float = Field(..., examples=[0.41])
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp of the last bar")


class RefreshResponse(BaseModel):
    refreshed: dict[str, str] = Field(
        ...,
        description="Symbol → 'ok' | 'missing' | 'stale'",
        examples=[{"BTC": "ok", "TSLA": "missing"}],
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe used by the Docker healthcheck."""
    try:
        forecaster = get_forecaster()
        model_id = forecaster.model_id
        loaded = forecaster.loaded
    except RuntimeError:
        model_id = "not initialised"
        loaded = False

    return HealthResponse(status="ok", model=model_id, loaded=loaded)


# ── /symbols ──────────────────────────────────────────────────────────────────

@router.get("/symbols", response_model=SymbolsResponse, tags=["ops"])
def symbols() -> SymbolsResponse:
    """List symbols that have parquet data loaded in memory."""
    store = get_data_store()
    return SymbolsResponse(symbols=store.available_symbols())


# ── /refresh ─────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=RefreshResponse, tags=["ops"])
def refresh() -> RefreshResponse:
    """Reload all parquet files from disk.

    Call this after re-running the data pipeline to pick up fresh data
    without restarting the service.
    """
    store = get_data_store()
    status = store.refresh()
    logger.info("Parquet refresh complete: {}", status)
    return RefreshResponse(refreshed=status)


# ── /regime ───────────────────────────────────────────────────────────────────

@router.get("/regime", response_model=RegimeResponse, tags=["regime"])
def regime(
    symbol: str = Query(
        ...,
        description="Symbol to classify.  Supported: BTC, TSLA.",
        examples=["BTC"],
    ),
    lookback: int = Query(
        default=512,
        ge=64,
        le=4096,
        description="Number of recent bars to use as Chronos context window.",
    ),
) -> RegimeResponse:
    """Run Chronos-T5 zero-shot forecasting and classify the market regime.

    The endpoint:
    1. Retrieves the cached DataFrame for ``symbol``.
    2. Extracts the last ``lookback`` close prices as Chronos context.
    3. Runs a 10-step probabilistic forecast.
    4. Combines the forecast spread signal with the Hurst DFA signal.
    5. Returns the regime label, confidence, forecast price range, and
       diagnostics.

    Raises
    ------
    400 — if the symbol is not supported or its parquet is not loaded.
    500 — if Chronos inference fails.
    """
    store = get_data_store()

    # ── Validate symbol ───────────────────────────────────────────────────────
    if not store.is_supported(symbol):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Symbol '{symbol}' is not supported.  "
                f"Supported symbols: {list(store.available_symbols())}."
            ),
        )

    df = store.get_dataframe(symbol)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No data found for symbol {symbol}.  "
                "Run the data pipeline first."
            ),
        )

    # ── Extract context window ────────────────────────────────────────────────
    close_series = df["close"].dropna()
    context = close_series.tail(lookback).values.astype(np.float64)

    if len(context) < 32:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient data for {symbol}: only {len(context)} clean "
                f"close bars available (minimum 32 required)."
            ),
        )

    # ── Extract ATR and Hurst ─────────────────────────────────────────────────
    try:
        current_atr = get_latest_atr(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        hurst_value = get_latest_hurst(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Chronos inference ─────────────────────────────────────────────────────
    forecaster = get_forecaster()
    if not forecaster.loaded:
        raise HTTPException(
            status_code=503,
            detail="Chronos model is not yet loaded.  Retry in a moment.",
        )

    logger.info(
        "Running regime classification for {} (context_len={} lookback_requested={})",
        symbol,
        len(context),
        lookback,
    )

    try:
        forecast = forecaster.predict(context=context, prediction_length=10)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chronos inference failed for {}: {}", symbol, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Chronos inference failed: {exc}",
        ) from exc

    # ── Regime classification ─────────────────────────────────────────────────
    result = classify_regime(
        q10=forecast.q10,
        q90=forecast.q90,
        current_atr=current_atr,
        hurst_value=hurst_value,
    )

    logger.info(
        "Regime: symbol={} regime={} confidence={:.4f} "
        "spread_atr_ratio={:.4f} hurst={:.4f}",
        symbol,
        result.regime.value,
        result.confidence,
        result.spread_atr_ratio,
        result.hurst,
    )

    # ── Timestamp from last bar index ─────────────────────────────────────────
    timestamp_str = _last_bar_timestamp(df)

    return RegimeResponse(
        symbol=symbol,
        regime=result.regime.value,
        confidence=result.confidence,
        forecast_range=ForecastRange(
            low=round(result.forecast_low, 2),
            high=round(result.forecast_high, 2),
        ),
        hurst=result.hurst,
        timestamp=timestamp_str,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _last_bar_timestamp(df: pd.DataFrame) -> str:
    """Return the last index value as a UTC ISO-8601 string."""

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        ts: pd.Timestamp = idx[-1]
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fallback: return current UTC time
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
