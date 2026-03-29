"""
src/piccolo/config_live.py

Live trading symbol universe configuration.

LIVE_SYMBOLS defines which tickers receive daily EOD price ingestion,
options chain snapshots, and live signal generation.

MIN_HISTORY_DAYS is the minimum number of trading days of price history
required before features can be computed for a symbol. This must be
sufficient to cover SMA-200 (200 days) plus the longest rolling return
window used as a feature (e.g., 60-day returns).
"""

# Symbols receiving live data ingestion and signal generation
LIVE_SYMBOLS = ["SPY", "QQQ", "VOO", "AAPL"]

# Minimum price history required to compute all features
MIN_HISTORY_DAYS = 270  # Enough for SMA-200 + 60-day returns
