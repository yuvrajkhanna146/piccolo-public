"""
src/piccolo — Piccolo ML Options Strategy

Core package containing:
    config_strategy         Strategy hyperparameters (thresholds, horizons, XGBoost params)
    config_live             Live trading symbol universe
    ml_signal_engine        Feature loading, label construction, walk-forward training, ensemble
    run_daily_pipeline      Daily pipeline orchestrator (Steps 1–4)
    td_options_snapshot     Theta Data: daily options chain ingestion
    eod_prices_td           Theta Data: EOD price ingestion
    notify_email            Signal email notification

    legacy/                 IBKR-based scripts (superseded by Theta Data, retained for reference)
"""
