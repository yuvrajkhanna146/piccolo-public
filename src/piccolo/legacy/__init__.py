"""
src/piccolo/legacy — IBKR-based data ingestion scripts (superseded by Theta Data)

These scripts used Interactive Brokers TWS/Gateway for live data collection.
They have been replaced by the Theta Data pipeline (td_options_snapshot.py,
eod_prices_td.py) but are retained here for reference and for users who
prefer IBKR as their data source.

    bootstrap_eod_prices_ibkr   One-time historical price backfill via IBKR
    eod_prices_daily_ibkr       Nightly EOD price top-up via IBKR
    ibkr_options_snapshot       Daily options chain capture via IBKR
"""
