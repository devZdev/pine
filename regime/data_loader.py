"""
regime/data_loader.py
=====================
Parquet loading, in-memory caching, and symbol→file mapping.

Responsibilities
----------------
- Maintain a registry that maps short symbol names (``BTC``, ``TSLA``) to
  their Parquet file paths under ``DATA_DIR``.
- Load and cache all available DataFrames at startup.
- Provide a ``refresh()`` method that reloads from disk (called by
  ``POST /refresh``).
- Expose ``get_dataframe()`` for route handlers to retrieve cached data.

The module deliberately has **no dependency** on FastAPI so it can be unit-
tested in isolation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Symbol → filename mapping
# ---------------------------------------------------------------------------

# Keys are the short symbol names accepted by the API.
# Values are filenames relative to ``DATA_DIR``.
SYMBOL_FILE_MAP: dict[str, str] = {
    "BTC": "BTC_USD_5m.parquet",
    "TSLA": "TSLA_5m.parquet",
}


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

class DataStore:
    """In-memory store for per-symbol DataFrames loaded from Parquet.

    Parameters
    ----------
    data_dir : str | Path | None
        Root directory that contains the Parquet files.  Defaults to the
        value of the ``DATA_DIR`` environment variable, falling back to
        ``data/raw``.
    """

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        self._data_dir: Path = Path(
            data_dir
            or os.environ.get("DATA_DIR", "data/raw")
        )
        # symbol → DataFrame (only present when successfully loaded)
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """Resolved data directory."""
        return self._data_dir

    @property
    def loaded_symbols(self) -> list[str]:
        """Symbols for which data is currently cached."""
        return list(self._cache.keys())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Attempt to load every known symbol from disk.

        Symbols whose Parquet files do not exist are skipped with a warning.
        Existing cache entries are replaced on success.
        """
        logger.info(
            "Loading Parquet files from '{}' …", self._data_dir
        )
        for symbol, filename in SYMBOL_FILE_MAP.items():
            path = self._data_dir / filename
            df = _load_parquet_safe(symbol, path)
            if df is not None:
                self._cache[symbol] = df
                logger.info(
                    "Loaded {} — {} rows, columns: {}",
                    symbol,
                    len(df),
                    list(df.columns),
                )
            else:
                if symbol in self._cache:
                    # Keep stale data rather than removing it on a failed reload
                    logger.warning(
                        "Reload of {} failed — keeping stale cache.", symbol
                    )
                else:
                    logger.warning(
                        "Symbol {} not loaded (file not found or unreadable).",
                        symbol,
                    )

    def refresh(self) -> dict[str, str]:
        """Reload all parquet files from disk.

        Returns
        -------
        dict[str, str]
            ``{symbol: "ok" | "missing" | "stale"}`` status per symbol.
        """
        logger.info("Refreshing Parquet cache …")
        status: dict[str, str] = {}
        for symbol, filename in SYMBOL_FILE_MAP.items():
            path = self._data_dir / filename
            df = _load_parquet_safe(symbol, path)
            if df is not None:
                self._cache[symbol] = df
                status[symbol] = "ok"
                logger.info("Refreshed {}.", symbol)
            else:
                status[symbol] = "stale" if symbol in self._cache else "missing"
                logger.warning(
                    "Refresh of {} failed — status: {}.", symbol, status[symbol]
                )
        return status

    def get_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return the cached DataFrame for ``symbol``, or ``None``.

        Parameters
        ----------
        symbol : str
            Short symbol name, e.g. ``"BTC"``.

        Returns
        -------
        pd.DataFrame | None
            The cached DataFrame, or ``None`` if not loaded.
        """
        return self._cache.get(symbol)

    def is_supported(self, symbol: str) -> bool:
        """Return ``True`` if the symbol is in the known registry."""
        return symbol in SYMBOL_FILE_MAP

    def available_symbols(self) -> list[str]:
        """Return symbols that are both known and currently cached."""
        return [s for s in SYMBOL_FILE_MAP if s in self._cache]


# ---------------------------------------------------------------------------
# Helper: column extraction
# ---------------------------------------------------------------------------

def extract_context(
    df: pd.DataFrame,
    lookback: int = 512,
    column: str = "close",
) -> tuple[float, float]:  # (close_array as np.ndarray placeholder)
    """This is just a docstring anchor; the real extraction lives in router.py.

    The router calls ``df[column].dropna().tail(lookback).values`` directly
    to keep the data flow transparent.  This module provides the
    ``get_latest_atr`` and ``get_latest_hurst`` helpers below.
    """
    ...  # pragma: no cover


def get_latest_atr(df: pd.DataFrame) -> float:
    """Return the most recent non-NaN ``atr_14`` value.

    Raises
    ------
    ValueError
        If the ``atr_14`` column is absent or entirely NaN.
    """
    if "atr_14" not in df.columns:
        raise ValueError("DataFrame is missing the 'atr_14' column.")
    series = df["atr_14"].dropna()
    if series.empty:
        raise ValueError("'atr_14' column contains no non-NaN values.")
    return float(series.iloc[-1])


def get_latest_hurst(df: pd.DataFrame) -> float:
    """Return the most recent non-NaN ``hurst_dfa`` value.

    Raises
    ------
    ValueError
        If the ``hurst_dfa`` column is absent or entirely NaN.
    """
    if "hurst_dfa" not in df.columns:
        raise ValueError("DataFrame is missing the 'hurst_dfa' column.")
    series = df["hurst_dfa"].dropna()
    if series.empty:
        raise ValueError("'hurst_dfa' column contains no non-NaN values.")
    return float(series.iloc[-1])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_parquet_safe(symbol: str, path: Path) -> Optional[pd.DataFrame]:
    """Load a Parquet file, returning ``None`` on any error."""
    if not path.exists():
        logger.warning(
            "Parquet file for {} not found at '{}'.", symbol, path
        )
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("Parquet for {} loaded but is empty.", symbol)
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read Parquet for {}: {}", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_data_store: Optional[DataStore] = None


def get_data_store() -> DataStore:
    """Return the module-level DataStore singleton."""
    if _data_store is None:
        raise RuntimeError(
            "DataStore has not been initialised.  "
            "Ensure the FastAPI lifespan startup has completed."
        )
    return _data_store


def set_data_store(store: DataStore) -> None:
    """Set the module-level singleton (called by lifespan startup)."""
    global _data_store
    _data_store = store
