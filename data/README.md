# Data

This directory is intentionally empty. The strategy requires:

1. **CBOE historical options data** — EOD quotes (10+ years of SPY options chains)
2. **EOD prices** — Daily closes via Theta Data terminal (or IBKR for legacy setup)

Data is stored in DuckDB databases configured via environment variables.
See `.env.example` for the required paths.

The ingestion scripts in `src/piccolo/` handle data loading:

**Theta Data (current):**
- `eod_prices_td.py` — Daily EOD price ingestion; seeds 3 years of history on first run
- `td_options_snapshot.py` — Daily options chain snapshot

**IBKR (legacy, superseded):**
- `legacy/bootstrap_eod_prices_ibkr.py` — One-time historical price backfill
- `legacy/eod_prices_daily_ibkr.py` — Nightly EOD price top-up
- `legacy/ibkr_options_snapshot.py` — Daily options chain capture
