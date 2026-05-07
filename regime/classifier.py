"""
regime/classifier.py
====================
Regime classification logic combining two independent signals:

1. **Chronos forecast spread** — normalised spread of the probabilistic
   forecast relative to the current ATR.  A wide spread indicates the model
   sees high directional uncertainty (TRENDING); a narrow spread indicates
   mean-reversion territory.

2. **Hurst exponent (DFA)** — taken directly from the Phase 1 feature column.
   ``hurst < 0.45`` → MEAN_REVERTING; ``hurst > 0.55`` → TRENDING.

The two signals are combined into a single ``RegimeResult`` that carries the
regime label, a confidence score, and supporting diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"


# Signal-level intermediate label (includes NEUTRAL for Hurst in grey zone)
_HurstSignal = Literal["TRENDING", "MEAN_REVERTING", "NEUTRAL"]
_SpreadSignal = Literal["TRENDING", "MEAN_REVERTING"]


# ---------------------------------------------------------------------------
# Thresholds (module-level constants for easy tuning)
# ---------------------------------------------------------------------------

# Chronos spread / ATR threshold
SPREAD_ATR_THRESHOLD: float = 1.5

# Hurst thresholds
HURST_TRENDING_THRESHOLD: float = 0.55
HURST_MEAN_REVERTING_THRESHOLD: float = 0.45

# Base confidence when signals agree / disagree
CONFIDENCE_AGREE_BASE: float = 0.85
CONFIDENCE_DISAGREE_BASE: float = 0.55

# Weights in the combined confidence formula
CONFIDENCE_AGREE_BONUS: float = 0.30   # added when both signals agree
CONFIDENCE_HURST_BONUS: float = 0.15   # scaled by |hurst - 0.5| / 0.1


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Output of the regime classification step.

    Attributes
    ----------
    regime : Regime
        The classified market regime.
    confidence : float
        Score in [0, 1] expressing certainty of the classification.
    forecast_low : float
        Lowest price in the 10th-percentile forecast band (raw price level).
    forecast_high : float
        Highest price in the 90th-percentile forecast band (raw price level).
    hurst : float
        Most recent non-NaN Hurst DFA value used for classification.
    spread_signal : str
        Chronos signal before combination ("TRENDING" | "MEAN_REVERTING").
    hurst_signal : str
        Hurst signal before combination ("TRENDING" | "MEAN_REVERTING" | "NEUTRAL").
    spread_atr_ratio : float
        ``forecast_spread / atr_14`` — diagnostic metric.
    """

    regime: Regime
    confidence: float
    forecast_low: float
    forecast_high: float
    hurst: float
    spread_signal: str
    hurst_signal: str
    spread_atr_ratio: float


# ---------------------------------------------------------------------------
# Public classifier function
# ---------------------------------------------------------------------------

def classify_regime(
    q10: np.ndarray,
    q90: np.ndarray,
    current_atr: float,
    hurst_value: float,
) -> RegimeResult:
    """Classify the current market regime from Chronos quantiles + Hurst.

    Parameters
    ----------
    q10 : np.ndarray
        10th-percentile forecast for each of the next ``N`` steps.
    q90 : np.ndarray
        90th-percentile forecast for each of the next ``N`` steps.
    current_atr : float
        Current ATR-14 value (last non-NaN row).
    hurst_value : float
        Most recent non-NaN ``hurst_dfa`` value.

    Returns
    -------
    RegimeResult
        Combined regime label with confidence and diagnostics.

    Raises
    ------
    ValueError
        If ``q10`` and ``q90`` have different shapes, or if ``current_atr``
        is not a positive finite number.
    """
    if q10.shape != q90.shape:
        raise ValueError(
            f"q10 and q90 must have the same shape; "
            f"got {q10.shape} vs {q90.shape}."
        )
    if not (np.isfinite(current_atr) and current_atr > 0):
        raise ValueError(
            f"current_atr must be a positive finite number; got {current_atr!r}."
        )
    if not np.isfinite(hurst_value):
        raise ValueError(f"hurst_value must be finite; got {hurst_value!r}.")

    # ── Signal 1: Chronos forecast spread ─────────────────────────────────────
    per_step_spread: np.ndarray = q90 - q10
    mean_spread: float = float(np.mean(per_step_spread))
    spread_atr_ratio: float = mean_spread / current_atr

    spread_signal: _SpreadSignal = (
        "TRENDING" if spread_atr_ratio > SPREAD_ATR_THRESHOLD else "MEAN_REVERTING"
    )

    logger.debug(
        "Spread signal: mean_spread={:.4f} atr={:.4f} ratio={:.4f} → {}",
        mean_spread,
        current_atr,
        spread_atr_ratio,
        spread_signal,
    )

    # ── Signal 2: Hurst exponent ──────────────────────────────────────────────
    hurst_signal: _HurstSignal
    if hurst_value < HURST_MEAN_REVERTING_THRESHOLD:
        hurst_signal = "MEAN_REVERTING"
    elif hurst_value > HURST_TRENDING_THRESHOLD:
        hurst_signal = "TRENDING"
    else:
        hurst_signal = "NEUTRAL"

    logger.debug(
        "Hurst signal: hurst={:.4f} → {}",
        hurst_value,
        hurst_signal,
    )

    # ── Combination logic ──────────────────────────────────────────────────────
    # The Hurst signal is only non-neutral when it agrees or disagrees clearly.
    # When neutral we treat it as "not disagreeing" with the Chronos signal.

    both_agree: bool = (
        hurst_signal == spread_signal
    )  # NEUTRAL never equals TRENDING or MEAN_REVERTING

    if both_agree:
        final_regime = Regime(spread_signal)
        logger.debug("Signals agree → regime={}", final_regime)
    else:
        # Chronos is fresher; use its signal
        final_regime = Regime(spread_signal)
        logger.debug(
            "Signals disagree (spread={}, hurst={}) → defaulting to Chronos → {}",
            spread_signal,
            hurst_signal,
            final_regime,
        )

    # ── Confidence formula ─────────────────────────────────────────────────────
    # confidence = 0.55
    #            + 0.30 * both_agree
    #            + 0.15 * min(1.0, |hurst - 0.5| / 0.1)
    hurst_deviation: float = min(1.0, abs(hurst_value - 0.5) / 0.1)
    confidence: float = (
        CONFIDENCE_DISAGREE_BASE
        + CONFIDENCE_AGREE_BONUS * float(both_agree)
        + CONFIDENCE_HURST_BONUS * hurst_deviation
    )
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    logger.debug(
        "Confidence: base=0.55 agree_bonus={:.2f} hurst_bonus={:.4f} → {:.4f}",
        CONFIDENCE_AGREE_BONUS * float(both_agree),
        CONFIDENCE_HURST_BONUS * hurst_deviation,
        confidence,
    )

    return RegimeResult(
        regime=final_regime,
        confidence=round(confidence, 4),
        forecast_low=float(np.min(q10)),
        forecast_high=float(np.max(q90)),
        hurst=round(float(hurst_value), 6),
        spread_signal=spread_signal,
        hurst_signal=hurst_signal,
        spread_atr_ratio=round(spread_atr_ratio, 6),
    )
