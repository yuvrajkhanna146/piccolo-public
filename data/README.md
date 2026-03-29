# Data

This directory is intentionally empty. The strategy requires:

1. **CBOE historical options data** — EOD quotes (10+ years of SPY options chains)
2. **IBKR EOD prices** — Daily OHLCV via the Interactive Brokers API

Data is stored in DuckDB databases configured via environment variables.
See `.env.example` for the required paths.

The ingestion scripts in `src/piccolo/` handle data loading:
- `bootstrap_eod_prices_ibkr.py` — One-time historical price backfill
- `eod_prices_daily_ibkr.py` — Nightly EOD price top-up
- `ibkr_options_snapshot.py` — Daily options chain capture
