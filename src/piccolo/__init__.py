"""
src/piccolo — Piccolo ML Options Strategy

Core package containing:
    config_strategy     Strategy hyperparameters (thresholds, horizons, XGBoost params)
    config_live         Live trading symbol universe
    ml_signal_engine    Feature loading, label construction, walk-forward training, ensemble
    bootstrap_eod_prices_ibkr   One-time IBKR historical price backfill
    eod_prices_daily_ibkr       Nightly EOD price top-up
    ibkr_options_snapshot       Daily options chain capture
"""
