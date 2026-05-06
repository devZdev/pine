"""
backtest
========
Institutional-grade cash-secured put backtester with walk-forward optimization.

Modules
-------
options_math      Black-Scholes pricing, realized vol, strike solver
signal_generator  Glass Box entry/exit signals (vectorized)
kelly_sizer       Half-Kelly position sizing
wfo_engine        Walk-forward fold runner and parameter optimizer
performance       Calmar, Sortino, Max Drawdown, benchmark comparison
simulator         Trade-by-trade options P&L simulation
backtester        Orchestrates all modules, runs full backtest
"""
