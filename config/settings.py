"""
config/settings.py

Global environment-based configuration for the Piccolo strategy.

All paths are resolved from environment variables defined in .env (see .env.example).
No hardcoded paths should appear here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# DuckDB with daily options open-interest summary.
# Contains aggregated options OI tables used by daily OI analysis scripts.
DUCKDB_PATH_DAILY = os.getenv("DUCKDB_PATH_DAILY_OI")

# Main historical EOD DuckDB.
# Stores CBOE historical options data and the enriched features_spy_latest table
# used by ml_signal_engine for walk-forward training.
DUCKDB_PATH_HIST = os.getenv("DUCKDB_PATH_HIST_EOD")

# Raw CBOE DuckDB.
# Read-only source of raw options_cboe_eod data; used during feature enrichment.
DUCKDB_PATH_RAW = os.getenv("DUCKDB_PATH_HIST_RAW")

# Root folder containing raw IBKR export files (CSV / raw bars).
# Used by ingest scripts that populate historical DuckDB tables.
RAW_IBKR_PATH = os.getenv("RAW_IBKR_PATH")

# Root folder containing raw CBOE zip files.
# Used by data extraction scripts that load monthly options CSVs into DuckDB.
RAW_CBOE_PATH = os.getenv("RAW_CBOE_PATH")

# Live DuckDB for daily EOD prices and signals.
# Written to by eod_prices_td.py; read by ml_signal_inference.py.
DUCKDB_PATH_LIVE = os.getenv("DUCKDB_PATH_LIVE")

# Live options-chain DuckDB for daily snapshots.
# Written to by td_options_snapshot.py; stores per-trade-date option chains
# used for live ML feature generation.
DUCKDB_PATH_LIVE_OPTIONS = os.getenv("DUCKDB_PATH_LIVE_OPTIONS")

# Directory for trained ensemble SPY model artifacts.
# ml_signal_engine.py saves/loads XGBoost models, scalers, and config JSON here.
ENSEMBLE_ARTIFACTS_DIR = os.getenv("ENSEMBLE_ARTIFACTS_DIR")
