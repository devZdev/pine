"""
storage.py
==========
Parquet read/write helpers.

All files are saved with Snappy compression via PyArrow.  A FastParquet
fallback is attempted if PyArrow is unavailable (unlikely in practice).

Filename convention:  data/raw/{symbol}_{timeframe}.parquet
  e.g.  data/raw/BTC_USD_1m.parquet
        data/raw/TSLA_1m.parquet
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR: Path = Path("data") / "raw"
COMPRESSION: Literal["snappy"] = "snappy"

# PyArrow is the primary engine; fastparquet is the fallback.
try:
    import pyarrow  # noqa: F401
    _PREFERRED_ENGINE: Literal["pyarrow", "fastparquet"] = "pyarrow"
except ImportError:  # pragma: no cover
    _PREFERRED_ENGINE = "fastparquet"
    logger.warning("PyArrow not found; falling back to fastparquet.")


# ── Public API ────────────────────────────────────────────────────────────────

def raw_path(symbol: str, timeframe: str, base_dir: Path | str = DEFAULT_RAW_DIR) -> Path:
    """Return the canonical parquet path for a symbol/timeframe pair.

    Parameters
    ----------
    symbol:
        e.g. ``"BTC_USD"`` or ``"TSLA"``
    timeframe:
        e.g. ``"1m"`` or ``"5m"``
    base_dir:
        Root directory.  Defaults to ``data/raw``.

    Returns
    -------
    Path
        e.g. ``data/raw/BTC_USD_1m.parquet``
    """
    base_dir = Path(base_dir)
    return base_dir / f"{symbol}_{timeframe}.parquet"


def save_parquet(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    base_dir: Path | str = DEFAULT_RAW_DIR,
) -> Path:
    """Persist *df* to a Snappy-compressed parquet file.

    The DataFrame index is written as a column named ``timestamp`` so that the
    time axis survives a round-trip through both PyArrow and fastparquet.

    Parameters
    ----------
    df:
        OHLCV (+ feature) DataFrame with a UTC ``DatetimeIndex``.
    symbol:
        Ticker label used in the filename.
    timeframe:
        Candle width label used in the filename.
    base_dir:
        Directory where the file is written.  Created if it does not exist.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_path(symbol, timeframe, base_dir)

    # Ensure the index is tz-aware UTC before serialising
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df = df.copy()
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"

    df.to_parquet(
        dest,
        engine=_PREFERRED_ENGINE,
        compression=COMPRESSION,
        index=True,
    )

    file_size_mb = dest.stat().st_size / (1024 ** 2)
    logger.info(
        "Saved {} rows → {} ({:.2f} MB, engine={}, compression={}).",
        len(df), dest, file_size_mb, _PREFERRED_ENGINE, COMPRESSION,
    )
    return dest.resolve()


def load_parquet(
    symbol: str,
    timeframe: str,
    base_dir: Path | str = DEFAULT_RAW_DIR,
) -> pd.DataFrame | None:
    """Load a previously saved parquet file.

    Returns ``None`` if the file does not exist.

    Parameters
    ----------
    symbol:
        Ticker label.
    timeframe:
        Candle width label.
    base_dir:
        Directory to search.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with a UTC ``DatetimeIndex`` named ``timestamp``,
        or ``None`` if no file found.
    """
    path = raw_path(symbol, timeframe, base_dir)
    if not path.exists():
        logger.debug("No existing parquet found at {}.", path)
        return None

    df = pd.read_parquet(path, engine=_PREFERRED_ENGINE)

    # Restore the DatetimeIndex
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.sort_index(inplace=True)
    logger.info("Loaded {} rows from {}.", len(df), path)
    return df


def checkpoint_save(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    base_dir: Path | str = DEFAULT_RAW_DIR,
) -> Path:
    """Identical to :func:`save_parquet` — a semantic alias used for partial
    progress saves triggered by ``KeyboardInterrupt``."""
    logger.warning(
        "Checkpoint save: {} rows for {}_{}.",
        len(df), symbol, timeframe,
    )
    return save_parquet(df, symbol, timeframe, base_dir)


def merge_and_deduplicate(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate *existing* and *new* DataFrames and drop duplicate timestamps.

    The *new* data takes precedence over *existing* on duplicates.

    Parameters
    ----------
    existing:
        Previously persisted data (may be ``None``).
    new:
        Freshly fetched data.

    Returns
    -------
    pd.DataFrame
        Merged, deduplicated, sorted DataFrame.
    """
    if existing is None or existing.empty:
        result = new.copy()
    else:
        result = pd.concat([existing, new])
        # Keep last occurrence (new data wins) for duplicate timestamps
        result = result[~result.index.duplicated(keep="last")]

    result.sort_index(inplace=True)
    return result
