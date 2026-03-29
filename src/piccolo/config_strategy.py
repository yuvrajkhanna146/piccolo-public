"""
src/piccolo/config_strategy.py

Piccolo strategy hyperparameters.

All configurable knobs for labelling, training, and signal generation.
Adjust these values based on your own backtesting and risk preferences.

Note: All values below are clearly marked as examples and do not represent
the author's live trading parameters.
"""

# ── Label / target definition ──────────────────────────────────────────────────
UP_THRESHOLD = 0.015          # Example: path-forward return to label as "Up"
DOWN_THRESHOLD = -0.015       # Example: path-forward return to label as "Down"
LABEL_HORIZON_DAYS = 5        # Example: forward-looking window in trading days

# ── Walk-forward training ──────────────────────────────────────────────────────
N_TRAIN_MONTHS = 36           # Example: rolling training window in months
N_TEST_MONTHS = 3             # Example: test window per fold in months

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "multi:softprob",
    "num_class": 3,
    "use_label_encoder": False,
}

# ── Signal confidence thresholds ───────────────────────────────────────────────
CONF_THRESHOLD_UP = 0.45      # Example: min ensemble P(Up) to fire long signal
CONF_THRESHOLD_DOWN = 0.45    # Example: min ensemble P(Down) to fire short signal

# ── Regime filters ─────────────────────────────────────────────────────────────
USE_FLAT_CLASS_FILTER = True        # Suppress trades when Flat is highest prob
USE_ABOVE_SMA200_FILTER = True      # Only go long when price > SMA-200
USE_VOL_REGIME_FILTER = True        # Suppress in extreme vol regimes

# ── Ensemble ───────────────────────────────────────────────────────────────────
ALPHA = 0.7                   # Example: exponential recency weight for fold ensemble
                              # Higher ALPHA → more equal weight across folds
                              # Lower ALPHA → concentrates weight on most recent fold
