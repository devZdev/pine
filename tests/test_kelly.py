"""
tests/test_kelly.py
===================
Phase 3 half-Kelly position sizing tests.

The production sizer takes (premium, strike, delta_target).  We construct
inputs that map to known textbook (p, b) values and verify the resulting
fractions match.

Textbook full-Kelly: f = p - (1-p)/b
Half-Kelly:          f/2
Cap at 25% portfolio allocation.
"""
from __future__ import annotations

import pytest

from backtest.kelly_sizer import (
    _MAX_POSITION_FRACTION,
    compute_kelly_fraction,
    compute_num_contracts,
    compute_position_size,
)

pytestmark = pytest.mark.phase3


def test_kelly_textbook_p_06_b_1():
    """Kelly(p=0.6, b=1) full = 0.2, half = 0.1.

    Map to (premium, strike, delta_target):
      p = 1 - delta_target = 0.6  → delta_target = 0.4
      b = premium / (strike - premium) = 1 → premium = strike / 2
    """
    delta_target = 0.4
    strike = 100.0
    premium = 50.0    # → b = 50 / 50 = 1.0
    half_kelly = compute_kelly_fraction(premium, strike, delta_target)
    # Full Kelly = 0.6 - 0.4/1 = 0.2 → half = 0.1
    assert half_kelly == pytest.approx(0.10, abs=1e-9)


def test_kelly_negative_returns_zero():
    """Negative full Kelly (unfavourable trade) → 0.

    Use p=0.4 (delta=0.6), b=1.0 → full Kelly = 0.4 - 0.6 = -0.2 < 0.
    """
    delta_target = 0.6
    strike = 100.0
    premium = 50.0
    f = compute_kelly_fraction(premium, strike, delta_target)
    assert f == 0.0


def test_kelly_capped_at_max_fraction():
    """Very favourable trade is capped at _MAX_POSITION_FRACTION (0.25)."""
    # delta = 0.05 (p=0.95), b huge → half-Kelly far above 0.25
    delta_target = 0.05
    strike = 100.0
    premium = 90.0   # b = 90/10 = 9
    f = compute_kelly_fraction(premium, strike, delta_target)
    # Full Kelly = 0.95 - 0.05/9 ≈ 0.9444 → half ≈ 0.4722, cap 0.25
    assert f == pytest.approx(_MAX_POSITION_FRACTION, abs=1e-9)


def test_kelly_premium_ge_strike_returns_zero():
    """Degenerate trade (premium >= strike) → 0."""
    assert compute_kelly_fraction(premium=110, strike=100, delta_target=0.2) == 0.0


def test_position_size_scales_linearly():
    """Position size doubles when portfolio doubles, holding (premium, strike) fixed."""
    pf1 = 100_000.0
    pf2 = 200_000.0
    p1 = compute_position_size(pf1, premium=2.0, strike=100.0, delta_target=0.20)
    p2 = compute_position_size(pf2, premium=2.0, strike=100.0, delta_target=0.20)
    assert p2 == pytest.approx(2.0 * p1, rel=1e-9)


def test_num_contracts_floors_to_int():
    """compute_num_contracts returns an integer >= 0."""
    n = compute_num_contracts(
        portfolio_value=10_000.0,
        premium=2.0,
        strike=100.0,
        underlying_price=100.0,
        contract_multiplier=1.0,
        delta_target=0.20,
    )
    assert isinstance(n, int)
    assert n >= 0
