"""
pipeline
========
Institutional-grade quant data ingestion and feature engineering pipeline.

Sub-modules
-----------
coinbase_ingestor  – async CCXT Coinbase BTC/USD OHLCV fetcher
alpaca_ingestor    – async Alpaca TSLA OHLCV fetcher
feature_engineer   – ATR, SMA, Bollinger Bands, Hurst (DFA)
storage            – Parquet read/write helpers (Snappy)
utils              – rate limiter, gap filler, logging setup
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pipeline")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "coinbase_ingestor",
    "alpaca_ingestor",
    "feature_engineer",
    "storage",
    "utils",
]
