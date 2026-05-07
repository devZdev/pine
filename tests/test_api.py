"""
tests/test_api.py
=================
Phase 2 FastAPI route tests using TestClient + mock ChronosForecaster.

We bypass the real lifespan (which would try to download Chronos) by manually
populating the DataStore singleton and registering a MockChronosForecaster.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from regime import data_loader as dl
from regime import model as rm
from regime.data_loader import DataStore

pytestmark = pytest.mark.phase2


@pytest.fixture
def app_client(synthetic_parquet_dir, mock_chronos_forecaster):
    """Build a TestClient with the singletons pre-populated.

    We import the FastAPI app *without* triggering its lifespan by using
    transport=None mode (TestClient enters the lifespan, so we set the
    singletons inside, before any request fires).
    """
    # Reset singletons first
    dl.set_data_store(DataStore(data_dir=synthetic_parquet_dir))
    dl.get_data_store().load_all()
    rm.set_forecaster(mock_chronos_forecaster)

    from main_regime import app

    # Override the lifespan with a no-op so the real Chronos download is skipped
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        # Re-set in case lifespan reset them
        dl.set_data_store(DataStore(data_dir=synthetic_parquet_dir))
        dl.get_data_store().load_all()
        rm.set_forecaster(mock_chronos_forecaster)
        yield

    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        yield client


def test_health_endpoint(app_client):
    """GET /health returns 200 with status=ok and the (mocked) model id."""
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert "model" in body


def test_symbols_endpoint(app_client):
    """GET /symbols lists loaded symbols."""
    r = app_client.get("/symbols")
    assert r.status_code == 200
    body = r.json()
    assert set(body["symbols"]) == {"BTC", "TSLA"}


def test_regime_endpoint_btc(app_client):
    """GET /regime?symbol=BTC returns a valid RegimeResponse."""
    r = app_client.get("/regime", params={"symbol": "BTC"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "BTC"
    assert body["regime"] in {"TRENDING", "MEAN_REVERTING"}
    assert 0.0 <= body["confidence"] <= 1.0
    fr = body["forecast_range"]
    assert "low" in fr and "high" in fr
    assert fr["high"] >= fr["low"]
    assert "hurst" in body
    assert "timestamp" in body


def test_regime_endpoint_unknown_symbol(app_client):
    """Unknown symbol → 400."""
    r = app_client.get("/regime", params={"symbol": "DOGE"})
    assert r.status_code == 400
    assert "not supported" in r.text.lower()


def test_regime_endpoint_missing_data(app_client, tmp_path, mock_chronos_forecaster):
    """Symbol in registry but no parquet → 400."""
    # Replace the data store with an empty one to simulate missing data
    (tmp_path / "raw").mkdir(exist_ok=True)
    empty_store = DataStore(data_dir=tmp_path / "raw")
    dl.set_data_store(empty_store)
    rm.set_forecaster(mock_chronos_forecaster)
    r = app_client.get("/regime", params={"symbol": "BTC"})
    assert r.status_code == 400


def test_refresh_endpoint(app_client):
    """POST /refresh returns the per-symbol status dict."""
    r = app_client.post("/refresh")
    assert r.status_code == 200
    body = r.json()
    assert "refreshed" in body
    assert set(body["refreshed"].keys()) == {"BTC", "TSLA"}
