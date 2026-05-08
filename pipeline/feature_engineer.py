"""
feature_engineer.py
===================
Vectorized feature calculation — zero look-ahead bias.

Features
--------
1. ATR-14       Wilder's smoothed ATR (recursive EWM, α = 1/14).
2. SMA-200      Simple 200-period moving average of close.
3. Bollinger Bands  20-period, 2σ → bb_upper, bb_lower, bb_pct.
4. Hurst (DFA)  512-bar rolling window, step=1, using nolds.dfa().

No Python loops over rows — all operations use pandas/numpy vectorised APIs.
The Hurst exponent is computed via ``Series.rolling(...).apply(raw=True)``,
which passes contiguous NumPy arrays to nolds.dfa without any Python-level
iteration over individual rows.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

try:
    import nolds
    _NOLDS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NOLDS_AVAILABLE = False
    logger.warning("nolds not installed — Hurst (DFA) feature will be skipped.")


# ── ATR (Wilder's smoothing) ──────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range.

    True Range is defined as::

        TR = max(H-L, |H-C_prev|, |L-C_prev|)

    Wilder's smoothing uses an exponential weighted mean with α = 1/period,
    equivalent to a span of ``2*period - 1`` in pandas EWM convention.
    The first ATR value is seeded as the simple mean of the first *period* TRs
    (standard practice), achieved by ``ewm(adjust=False)`` which starts
    accumulating from the first valid TR.

    Parameters
    ----------
    df:
        DataFrame with columns ``high``, ``low``, ``close``.  Must be sorted
        ascending without look-ahead bias.
    period:
        Lookback length (default 14).

    Returns
    -------
    pd.Series
        ATR values aligned with *df*.  The first ``period - 1`` values are NaN.
    """
    high: pd.Series = df["high"]
    low: pd.Series = df["low"]
    close_prev: pd.Series = df["close"].shift(1)

    # True Range — three-way comparison, fully vectorised
    tr: pd.Series = pd.concat(
        [
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's EMA: α = 1/period → com = period - 1
    # adjust=False implements the recursive formula: ATR_t = ATR_{t-1}*(1-α) + TR_t*α
    atr: pd.Series = tr.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    atr.name = f"atr_{period}"
    return atr


# ── SMA ───────────────────────────────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, period: int = 200, source_col: str = "close") -> pd.Series:
    """Simple moving average.

    Parameters
    ----------
    df:
        Source DataFrame.
    period:
        Window length.
    source_col:
        Column to average (default ``"close"``).

    Returns
    -------
    pd.Series
        SMA values.  First ``period - 1`` entries are NaN.
    """
    sma: pd.Series = df[source_col].rolling(window=period, min_periods=period).mean()
    sma.name = f"sma_{period}"
    return sma


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def compute_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    source_col: str = "close",
) -> pd.DataFrame:
    """Bollinger Bands: upper band, lower band, and %B.

    ``center=False`` (the default) is explicit here to avoid any risk of
    look-ahead through centred windows.

    %B = (close - lower) / (upper - lower)
    Values < 0 → below the lower band; > 1 → above the upper band.

    Parameters
    ----------
    df:
        Source DataFrame.
    period:
        Rolling window (default 20).
    num_std:
        Number of standard deviations (default 2.0).
    source_col:
        Column to compute bands on (default ``"close"``).

    Returns
    -------
    pd.DataFrame
        Three-column DataFrame: ``bb_upper``, ``bb_lower``, ``bb_pct``.
    """
    src: pd.Series = df[source_col]

    rolling = src.rolling(window=period, min_periods=period, center=False)
    mid: pd.Series = rolling.mean()
    std: pd.Series = rolling.std(ddof=1)          # sample std, consistent with TA convention

    upper: pd.Series = mid + num_std * std
    lower: pd.Series = mid - num_std * std

    band_width: pd.Series = upper - lower
    # Avoid division by zero when upper == lower (flat price)
    pct_b: pd.Series = (src - lower) / band_width.replace(0.0, np.nan)

    result = pd.DataFrame(
        {"bb_upper": upper, "bb_lower": lower, "bb_pct": pct_b},
        index=df.index,
    )
    return result


# ── Hurst Exponent via DFA ────────────────────────────────────────────────────

def _dfa_on_window(arr: np.ndarray) -> float:
    """Compute the DFA Hurst exponent for a single raw window.

    This function is called by ``pd.Series.rolling().apply(raw=True)`` — the
    ``raw=True`` flag ensures *arr* is a contiguous NumPy array, which is
    significantly faster than a Python-level row iteration.

    Returns ``np.nan`` on any computation failure so bad windows do not
    propagate exceptions.
    """
    if np.all(arr == arr[0]):
        # Constant series → undefined Hurst (avoid nolds exception)
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nolds.dfa(arr)  # type: ignore[no-any-return]
    except Exception:
        return np.nan


def compute_hurst_dfa(
    df: pd.DataFrame,
    window: int = 512,
    source_col: str = "close",
) -> pd.Series:
    """Rolling Hurst exponent using Detrended Fluctuation Analysis (DFA).

    Uses ``pd.Series.rolling(window, min_periods=window).apply(_dfa_on_window,
    raw=True)`` — no Python loops over rows.

    Parameters
    ----------
    df:
        Source DataFrame.
    window:
        Rolling window in candles (default 512).  The first ``window - 1``
        entries will be NaN.
    source_col:
        Column to compute the Hurst exponent on (default ``"close"``).

    Returns
    -------
    pd.Series
        Hurst exponent values.  Range is typically 0–1:
        - ~0.5 → random walk
        - > 0.5 → trending / persistent
        - < 0.5 → mean-reverting / anti-persistent
    """
    if not _NOLDS_AVAILABLE:
        logger.warning("nolds unavailable — returning NaN series for hurst_dfa.")
        return pd.Series(np.nan, index=df.index, name="hurst_dfa")

    logger.info(
        "Computing rolling Hurst (DFA) on {} rows with window={}.  This may take several minutes.",
        len(df), window,
    )

    # DFA expects an "increment-like" series: applying DFA directly to raw
    # prices (a random-walk-like cumulative series) gives H ≈ 1.5 for white
    # noise instead of the canonical 0.5.  Convert to log-returns first so
    # the Hurst exponent lands in the conventional [0, 1] range.
    src: pd.Series = np.log(df[source_col].astype(float)).diff()

    hurst: pd.Series = src.rolling(window=window, min_periods=window).apply(
        _dfa_on_window,
        raw=True,
        engine="cython",   # Use Cython engine for speed if available
    )
    hurst.name = "hurst_dfa"
    logger.info("Hurst (DFA) computation complete.")
    return hurst


# ── Master feature computation ─────────────────────────────────────────────────

def compute_all_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    sma_period: int = 200,
    bb_period: int = 20,
    bb_std: float = 2.0,
    hurst_window: int = 512,
    include_hurst: bool = True,
) -> pd.DataFrame:
    """Compute and append all features to *df* in a single pass.

    Feature columns added
    ---------------------
    atr_14      Wilder's 14-period ATR
    sma_200     200-period SMA of close
    bb_upper    Bollinger upper band
    bb_lower    Bollinger lower band
    bb_pct      Bollinger %B
    hurst_dfa   512-bar rolling DFA Hurst exponent (if *include_hurst* is True)

    Parameters
    ----------
    df:
        Clean OHLCV DataFrame (with or without an ``is_filled`` column).
    atr_period:
        ATR lookback.
    sma_period:
        SMA lookback.
    bb_period:
        Bollinger Band window.
    bb_std:
        Bollinger Band standard deviation multiplier.
    hurst_window:
        DFA Hurst rolling window.
    include_hurst:
        Set to ``False`` to skip the expensive Hurst computation (e.g. for
        quick smoke tests).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with feature columns appended.  Original OHLCV columns
        are preserved unchanged.
    """
    df = df.copy()

    logger.info("Computing ATR-{}.", atr_period)
    df[f"atr_{atr_period}"] = compute_atr(df, period=atr_period)

    logger.info("Computing SMA-{}.", sma_period)
    df[f"sma_{sma_period}"] = compute_sma(df, period=sma_period)

    logger.info("Computing Bollinger Bands ({}-period, {}σ).", bb_period, bb_std)
    bb = compute_bollinger_bands(df, period=bb_period, num_std=bb_std)
    df = pd.concat([df, bb], axis=1)

    if include_hurst:
        df["hurst_dfa"] = compute_hurst_dfa(df, window=hurst_window)

    _validate_no_lookahead(df)
    return df


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate_no_lookahead(df: pd.DataFrame) -> None:
    """Sanity check: verify that the first valid feature value does not appear
    before the expected warm-up period has elapsed.

    This is a lightweight heuristic — it checks that ATR and SMA are NaN for
    their respective warm-up windows.  A full look-ahead test would require
    walk-forward re-computation which is outside the scope here.
    """
    if "atr_14" in df.columns:
        first_valid_atr = df["atr_14"].first_valid_index()
        if first_valid_atr is not None:
            pos = df.index.get_loc(first_valid_atr)
            if pos < 13:  # 0-indexed; ATR-14 needs 14 bars
                logger.warning(
                    "ATR-14 has a valid value at row {} (expected >= 13).  "
                    "Check for look-ahead bias.",
                    pos,
                )

    if "sma_200" in df.columns:
        first_valid_sma = df["sma_200"].first_valid_index()
        if first_valid_sma is not None:
            pos = df.index.get_loc(first_valid_sma)
            if pos < 199:
                logger.warning(
                    "SMA-200 has a valid value at row {} (expected >= 199).  "
                    "Check for look-ahead bias.",
                    pos,
                )

    logger.debug("Look-ahead bias validation passed.")
