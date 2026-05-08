"""
tests/test_classifier.py
========================
Phase 2 classify_regime tests — pure function with explicit inputs.

Confidence formula spec:
    confidence = 0.55 + 0.30 * both_agree + 0.15 * min(1, |hurst-0.5|/0.1)
"""
from __future__ import annotations

import numpy as np
import pytest

from regime.classifier import (
    Regime,
    classify_regime,
)

pytestmark = pytest.mark.phase2


def _q_arrays(low: float, high: float, n: int = 10):
    return np.full(n, low), np.full(n, high)


def test_both_agree_trending_high_confidence():
    """Wide spread (>1.5*ATR) + high Hurst → TRENDING with confidence ≥ 0.85."""
    q10, q90 = _q_arrays(98, 110)   # spread = 12
    atr = 1.0                        # ratio = 12 → > 1.5 → TRENDING
    hurst = 0.65                     # > 0.55 → TRENDING
    res = classify_regime(q10, q90, atr, hurst)
    assert res.regime == Regime.TRENDING
    assert res.confidence >= 0.85


def test_both_agree_meanreverting_high_confidence():
    """Narrow spread + low Hurst → MEAN_REVERTING with confidence ≥ 0.85."""
    q10, q90 = _q_arrays(99.5, 100.5)  # spread = 1
    atr = 1.0                            # ratio = 1 → < 1.5 → MEAN_REVERTING
    hurst = 0.30                         # < 0.45 → MEAN_REVERTING
    res = classify_regime(q10, q90, atr, hurst)
    assert res.regime == Regime.MEAN_REVERTING
    assert res.confidence >= 0.85


def test_disagreement_uses_chronos_signal():
    """When Chronos and Hurst disagree, regime follows Chronos (the spread)."""
    q10, q90 = _q_arrays(95, 110)        # spread 15 → wide → TRENDING
    atr = 1.0
    hurst = 0.30                          # MEAN_REVERTING
    res = classify_regime(q10, q90, atr, hurst)
    assert res.regime == Regime.TRENDING
    # not both_agree → confidence = 0.55 + 0 + 0.15*min(1, 0.20/0.1)=0.55+0.30=cap at .15
    # actual = 0.55 + 0.0 + 0.15 = 0.70
    assert res.confidence == pytest.approx(0.70, abs=1e-4)


def test_neutral_hurst_treated_as_disagreement():
    """Hurst in (0.45, 0.55) → NEUTRAL → does not equal Chronos signal → disagree path."""
    q10, q90 = _q_arrays(99.5, 100.5)  # MEAN_REVERTING
    atr = 1.0
    hurst = 0.50
    res = classify_regime(q10, q90, atr, hurst)
    assert res.regime == Regime.MEAN_REVERTING
    # both_agree=False; hurst_dev = 0
    assert res.confidence == pytest.approx(0.55, abs=1e-4)
    assert res.hurst_signal == "NEUTRAL"


def test_hurst_nan_raises_valueerror():
    """NaN Hurst is rejected by the classifier (caller must handle)."""
    q10, q90 = _q_arrays(99.5, 100.5)
    with pytest.raises(ValueError, match="hurst_value must be finite"):
        classify_regime(q10, q90, current_atr=1.0, hurst_value=float("nan"))


def test_invalid_atr_raises():
    """ATR <= 0 or non-finite is rejected."""
    q10, q90 = _q_arrays(99.5, 100.5)
    with pytest.raises(ValueError):
        classify_regime(q10, q90, current_atr=0.0, hurst_value=0.4)
    with pytest.raises(ValueError):
        classify_regime(q10, q90, current_atr=float("nan"), hurst_value=0.4)


def test_shape_mismatch_raises():
    """q10 / q90 shape mismatch is rejected."""
    q10 = np.zeros(10)
    q90 = np.zeros(11)
    with pytest.raises(ValueError, match="same shape"):
        classify_regime(q10, q90, current_atr=1.0, hurst_value=0.4)


def test_confidence_formula_exact():
    """Confidence = 0.55 + 0.30*agree + 0.15*min(1, |h-0.5|/0.1) — verify exactly."""
    # agree=True, hurst=0.40 → |0.4-0.5|/0.1 = 1.0 → confidence = 0.55 + 0.30 + 0.15 = 1.0
    q10, q90 = _q_arrays(99.5, 100.5)  # MEAN_REVERTING
    res = classify_regime(q10, q90, current_atr=1.0, hurst_value=0.40)
    assert res.confidence == pytest.approx(1.0, abs=1e-4)

    # agree=True, hurst=0.45 (boundary, NEUTRAL when not strictly < 0.45) — actually 0.45 is NEUTRAL
    # so test agree=True with hurst=0.44: |0.44-0.5|/0.1 = 0.6
    res2 = classify_regime(q10, q90, current_atr=1.0, hurst_value=0.44)
    expected = 0.55 + 0.30 * 1 + 0.15 * 0.6
    assert res2.confidence == pytest.approx(expected, abs=1e-4)


def test_diagnostics_populated():
    """Forecast range, hurst, signals, and ratio are all reported."""
    q10, q90 = _q_arrays(99.5, 100.5)
    res = classify_regime(q10, q90, current_atr=1.0, hurst_value=0.30)
    assert res.forecast_low == pytest.approx(99.5)
    assert res.forecast_high == pytest.approx(100.5)
    assert res.hurst == pytest.approx(0.30)
    assert res.spread_signal in {"TRENDING", "MEAN_REVERTING"}
    assert res.hurst_signal in {"TRENDING", "MEAN_REVERTING", "NEUTRAL"}
    assert res.spread_atr_ratio == pytest.approx(1.0, abs=1e-6)
