"""
options_math.py
===============
Black-Scholes pricing, realized volatility, and strike solver for
cash-secured put backtesting.

All functions are stateless and operate on scalars or NumPy arrays.
No look-ahead bias — realized_vol uses only rolling past data via the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from loguru import logger


# ── Realized Volatility ───────────────────────────────────────────────────────

def compute_realized_vol(
    close: pd.Series,
    window: int = 30,
    trading_days: int = 252,
) -> pd.Series:
    """30-day rolling annualized realized volatility of log returns.

    Uses only past data (no centre=True).  The first ``window`` values will be
    NaN as expected for a causal estimator.

    Parameters
    ----------
    close:
        Daily close price series.
    window:
        Rolling window in calendar days (default 30).
    trading_days:
        Annualization factor (default 252).

    Returns
    -------
    pd.Series
        Annualized realized volatility, aligned with *close*.
    """
    log_returns: pd.Series = np.log(close / close.shift(1))
    realized_vol: pd.Series = (
        log_returns
        .rolling(window=window, min_periods=window)
        .std()
        * np.sqrt(trading_days)
    )
    realized_vol.name = "realized_vol"
    return realized_vol


# ── Black-Scholes Core ────────────────────────────────────────────────────────

def _bs_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes d1 component."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _bs_d2(d1: float, sigma: float, T: float) -> float:
    """Black-Scholes d2 component."""
    return d1 - sigma * np.sqrt(T)


def bs_put_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """Black-Scholes European put price.

    P = K·e^(-rT)·N(-d2) - S·N(-d1)

    Parameters
    ----------
    S:
        Spot price.
    K:
        Strike price.
    r:
        Risk-free rate (annualized, continuous compounding).
    sigma:
        Implied / realized volatility (annualized).
    T:
        Time to expiry in years.

    Returns
    -------
    float
        Put option price.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, r, sigma, T)
    d2 = _bs_d2(d1, sigma, T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(price, 0.0))


def bs_put_delta(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> float:
    """Black-Scholes put delta = N(d1) - 1.

    Returns a value in (-1, 0].

    Parameters
    ----------
    S:
        Spot price.
    K:
        Strike price.
    r:
        Risk-free rate (annualized).
    sigma:
        Volatility (annualized).
    T:
        Time to expiry in years.

    Returns
    -------
    float
        Put delta in the range (-1, 0].
    """
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = _bs_d1(S, K, r, sigma, T)
    return float(norm.cdf(d1) - 1.0)


# ── Strike Solver ─────────────────────────────────────────────────────────────

def solve_strike_for_delta(
    S: float,
    r: float,
    sigma: float,
    T: float,
    target_delta: float = -0.20,
) -> float:
    """Find strike K such that put delta equals *target_delta*.

    Uses ``scipy.optimize.brentq`` to solve:
        bs_put_delta(S, K, r, sigma, T) - target_delta = 0

    The search bracket is [S * 0.01, S * 2.0], which comfortably covers any
    reasonable OTM put strike.

    Parameters
    ----------
    S:
        Spot price.
    r:
        Risk-free rate (annualized).
    sigma:
        Volatility (annualized).
    T:
        Time to expiry in years.
    target_delta:
        Target put delta (negative, default -0.20 for 20-delta put).

    Returns
    -------
    float
        Strike price K.

    Raises
    ------
    ValueError
        If root cannot be bracketed (degenerate inputs).
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        logger.warning(
            "Degenerate inputs to strike solver: S={}, sigma={}, T={}. "
            "Returning 80% moneyness fallback.",
            S, sigma, T,
        )
        return S * 0.80

    def objective(K: float) -> float:
        return bs_put_delta(S, K, r, sigma, T) - target_delta

    K_low = S * 0.01
    K_high = S * 2.0

    try:
        strike = brentq(objective, K_low, K_high, xtol=1e-6, maxiter=200)
    except ValueError:
        # Fallback: use log-normal approximation for OTM strike
        logger.warning(
            "Brentq failed for S={:.4f}, sigma={:.4f}, T={:.4f}. "
            "Using moneyness approximation.",
            S, sigma, T,
        )
        # Approximate: ln(S/K) ≈ N^{-1}(delta+1)*sigma*sqrt(T) - (r + sigma^2/2)*T
        z = norm.ppf(target_delta + 1.0)
        log_moneyness = z * sigma * np.sqrt(T) - (r + 0.5 * sigma ** 2) * T
        strike = S * np.exp(-log_moneyness)

    return float(strike)


def vectorized_strike_solver(
    spot: pd.Series,
    realized_vol: pd.Series,
    r: float = 0.05,
    dte: int = 30,
    target_delta: float = -0.20,
) -> pd.Series:
    """Apply ``solve_strike_for_delta`` across a Series using ``pd.Series.apply``.

    This is called once at backtest setup on signal rows only — acceptable use
    of apply since it is not in the per-bar hot loop.

    Parameters
    ----------
    spot:
        Spot price at each signal bar.
    realized_vol:
        Realized volatility at each signal bar.
    r:
        Risk-free rate (annualized, constant).
    dte:
        Days to expiry (calendar days).
    target_delta:
        Target put delta (default -0.20).

    Returns
    -------
    pd.Series
        Strike prices aligned with *spot* index.
    """
    T = dte / 365.0

    def _solve(idx: int) -> float:
        return solve_strike_for_delta(
            S=float(spot.iloc[idx]),
            r=r,
            sigma=float(realized_vol.iloc[idx]),
            T=T,
            target_delta=target_delta,
        )

    result = pd.Series(
        [_solve(i) for i in range(len(spot))],
        index=spot.index,
        name="strike",
    )
    return result


def vectorized_put_premium(
    spot: pd.Series,
    strike: pd.Series,
    realized_vol: pd.Series,
    r: float = 0.05,
    dte: int = 30,
) -> pd.Series:
    """Compute Black-Scholes put premiums for a series of signal rows.

    Parameters
    ----------
    spot:
        Spot prices at entry.
    strike:
        Solved strike prices.
    realized_vol:
        Realized volatility at entry.
    r:
        Risk-free rate.
    dte:
        Days to expiry (calendar days).

    Returns
    -------
    pd.Series
        Put premiums aligned with *spot* index.
    """
    T = dte / 365.0

    premiums = np.array([
        bs_put_price(
            S=float(spot.iloc[i]),
            K=float(strike.iloc[i]),
            r=r,
            sigma=float(realized_vol.iloc[i]),
            T=T,
        )
        for i in range(len(spot))
    ])

    return pd.Series(premiums, index=spot.index, name="premium")
