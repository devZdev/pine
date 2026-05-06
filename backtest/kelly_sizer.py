"""
kelly_sizer.py
==============
Half-Kelly position sizing for cash-secured put selling.

Kelly fraction:
    p   = 1 - delta_target           (probability put expires worthless)
    b   = premium / (strike - premium) (reward-to-risk ratio)
    f   = p - (1-p)/b                 (full Kelly)
    f_half = f / 2                    (half-Kelly)

Position size = f_half × portfolio_value, capped at 25% per trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# Absolute cap on position size as fraction of portfolio
_MAX_POSITION_FRACTION: float = 0.25


def compute_kelly_fraction(
    premium: float,
    strike: float,
    delta_target: float = 0.20,
) -> float:
    """Compute the half-Kelly fraction for a single trade.

    Parameters
    ----------
    premium:
        Put premium received at entry.
    strike:
        Put strike price.
    delta_target:
        Absolute delta used for the trade (0.20 for 20-delta).
        This is used as the probability the put expires OTM.

    Returns
    -------
    float
        Half-Kelly fraction in [0, _MAX_POSITION_FRACTION].
        Returns 0.0 if Kelly is negative (unfavourable trade).
    """
    max_loss = strike - premium
    if max_loss <= 0:
        logger.warning(
            "Premium ({:.4f}) >= strike ({:.4f}): degenerate trade, sizing to 0.",
            premium, strike,
        )
        return 0.0

    # Probability of full profit (put expires worthless)
    p: float = 1.0 - delta_target   # = 0.80 for 20-delta

    # Reward-to-risk ratio
    b: float = premium / max_loss

    if b <= 0:
        return 0.0

    # Full Kelly
    q: float = 1.0 - p
    full_kelly: float = p - q / b

    if full_kelly <= 0:
        logger.debug(
            "Negative Kelly fraction ({:.4f}): trade not favourable, sizing to 0.",
            full_kelly,
        )
        return 0.0

    half_kelly: float = full_kelly / 2.0

    # Cap at maximum allowed position fraction
    capped: float = min(half_kelly, _MAX_POSITION_FRACTION)

    if capped < half_kelly:
        logger.debug(
            "Half-Kelly ({:.4f}) capped at max position fraction ({:.4f}).",
            half_kelly, _MAX_POSITION_FRACTION,
        )

    return capped


def compute_position_size(
    portfolio_value: float,
    premium: float,
    strike: float,
    delta_target: float = 0.20,
) -> float:
    """Compute dollar position size for a single trade.

    Position size represents the cash collateral required (i.e. the strike
    price per unit, scaled by the Kelly fraction of portfolio value).

    Parameters
    ----------
    portfolio_value:
        Current portfolio value in dollars.
    premium:
        Put premium received per unit.
    strike:
        Put strike price per unit.
    delta_target:
        Absolute put delta (default 0.20).

    Returns
    -------
    float
        Dollar amount of cash to secure (collateral).
    """
    fraction: float = compute_kelly_fraction(premium, strike, delta_target)
    position_size: float = fraction * portfolio_value
    return position_size


def compute_num_contracts(
    portfolio_value: float,
    premium: float,
    strike: float,
    underlying_price: float,
    contract_multiplier: float = 1.0,
    delta_target: float = 0.20,
) -> int:
    """Compute integer number of put contracts to sell.

    Cash-secured puts require collateral = strike * contract_multiplier per
    contract.  We size to the Kelly-derived dollar amount and divide by the
    per-contract collateral requirement, flooring to an integer.

    Parameters
    ----------
    portfolio_value:
        Current portfolio equity.
    premium:
        Put premium per underlying unit.
    strike:
        Strike price per underlying unit.
    underlying_price:
        Current spot price (used for scaling where contract_multiplier=1).
    contract_multiplier:
        Number of underlying units per contract (default 1 for crypto,
        100 for equity options).
    delta_target:
        Absolute delta (default 0.20).

    Returns
    -------
    int
        Number of contracts (>= 0).
    """
    collateral_per_contract: float = strike * contract_multiplier
    if collateral_per_contract <= 0:
        return 0

    dollar_size: float = compute_position_size(
        portfolio_value, premium, strike, delta_target
    )
    n_contracts: int = int(dollar_size / collateral_per_contract)
    return max(n_contracts, 0)


def vectorized_kelly_fractions(
    premiums: pd.Series,
    strikes: pd.Series,
    delta_target: float = 0.20,
) -> pd.Series:
    """Apply half-Kelly sizing across a Series of trades.

    Parameters
    ----------
    premiums:
        Put premiums at entry for each signal bar.
    strikes:
        Strike prices at entry for each signal bar.
    delta_target:
        Target delta (default 0.20).

    Returns
    -------
    pd.Series
        Half-Kelly fractions aligned with *premiums* index.
    """
    fractions = np.vectorize(compute_kelly_fraction)(
        premiums.values,
        strikes.values,
        delta_target,
    )
    return pd.Series(fractions, index=premiums.index, name="kelly_fraction")
