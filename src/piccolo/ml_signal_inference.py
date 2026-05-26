# src/piccolo/ml_signal_inference.py
"""
ML signal inference — loads the saved ensemble and generates today's signal.

Reads the most recent row from features_spy_latest, applies the full fold
ensemble (weighted predict_proba), and writes the result to live_signals in
DUCKDB_PATH_LIVE. Re-running is safe: any existing row for the same date and
symbol is replaced before inserting.

Run after ml_signal_engine has produced artifacts:

    python -m src.piccolo.ml_signal_inference

Prerequisites:
    - Ensemble artifacts in ENSEMBLE_ARTIFACTS_DIR (produced by ml_signal_engine)
    - features_spy_latest populated in DUCKDB_PATH_HIST
    - All .env variables configured (see .env.example)
"""

import os
from datetime import date
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config.settings import DUCKDB_PATH_HIST, DUCKDB_PATH_LIVE
from src.piccolo.ml_signal_engine import load_ensemble
from src.piccolo.config_strategy import (
    CONF_THRESHOLD_UP,
    CONF_THRESHOLD_DOWN,
    USE_FLAT_CLASS_FILTER,
    USE_ABOVE_SMA200_FILTER,
    USE_VOL_REGIME_FILTER,
)

load_dotenv()

ARTIFACTS_DIR = os.getenv("ENSEMBLE_ARTIFACTS_DIR", "models/ensemble_spy_latest")
SYMBOL = "SPY"

# Optional context columns written to live_signals when present in the feature table.
# Used by notify_email.py for the signal report; NULL if not in features_spy_latest.
_CONTEXT_COLS = [
    "above_sma200", "vol_regime", "sma_200",
    "ret_20d", "ret_60d", "vol_20d", "vol_5d", "daily_ret",
]

_CREATE_LIVE_SIGNALS = """
    CREATE TABLE IF NOT EXISTS live_signals (
        quote_date     DATE,
        symbol         VARCHAR,
        signal_dir     INTEGER,
        proba_up_ens   DOUBLE,
        proba_flat_ens DOUBLE,
        proba_down_ens DOUBLE,
        above_sma200   DOUBLE,
        vol_regime     DOUBLE,
        sma_200        DOUBLE,
        ret_20d        DOUBLE,
        ret_60d        DOUBLE,
        vol_20d        DOUBLE,
        vol_5d         DOUBLE,
        daily_ret      DOUBLE
    )
"""


def _ensemble_proba(
    row: pd.Series,
    feature_cols: list,
    fold_models: dict,
    fold_scalers: dict,
    fold_weight_map: dict,
) -> np.ndarray:
    """
    Apply all fold models to a single feature vector.
    Returns weighted probability array [p_down, p_flat, p_up].
    """
    x = row[feature_cols].fillna(0.0).astype(float).values.reshape(1, -1)
    p_acc, w_acc = np.zeros(3), 0.0

    for fid, model in fold_models.items():
        w = fold_weight_map.get(fid, 0.0)
        if w <= 0:
            continue
        x_scaled = fold_scalers[fid].transform(x)
        p_acc += w * model.predict_proba(x_scaled)[0]
        w_acc += w

    return p_acc / w_acc if w_acc > 0 else np.full(3, 1 / 3)


def _decide_signal(proba: np.ndarray, context: dict) -> int:
    """
    Apply confidence thresholds and regime filters to ensemble probabilities.
    proba: [p_down, p_flat, p_up]
    Returns: 1 (Up), -1 (Down), 0 (Flat).
    """
    p_down, p_flat, p_up = proba

    if USE_FLAT_CLASS_FILTER and p_flat >= p_up and p_flat >= p_down:
        return 0

    if p_up >= CONF_THRESHOLD_UP and p_up >= p_down:
        signal = 1
    elif p_down >= CONF_THRESHOLD_DOWN and p_down >= p_up:
        signal = -1
    else:
        signal = 0

    if USE_ABOVE_SMA200_FILTER and context.get("above_sma200") is not None:
        if signal == 1 and context["above_sma200"] != 1:
            signal = 0

    if USE_VOL_REGIME_FILTER and context.get("vol_regime") is not None:
        if context["vol_regime"] == 0:
            signal = 0

    return signal


def run_inference(trade_date: Optional[date] = None) -> None:
    """
    Load saved ensemble, run on latest SPY features, write signal to live_signals.
    trade_date defaults to the quote_date of the most recent features row.
    """
    print(f"Loading ensemble from: {ARTIFACTS_DIR}")
    feature_cols, fold_models, fold_scalers, fold_weight_map = load_ensemble(ARTIFACTS_DIR)
    print(f"  {len(fold_models)} fold models | {len(feature_cols)} features")

    con = duckdb.connect(DUCKDB_PATH_HIST, read_only=True)
    row_df = con.execute(
        "SELECT * FROM features_spy_latest ORDER BY quote_date DESC LIMIT 1"
    ).df()
    con.close()

    if row_df.empty:
        raise RuntimeError(
            "features_spy_latest is empty — has td_options_snapshot.py and "
            "eod_prices_td.py been run?"
        )

    row = row_df.iloc[0]
    row_date = pd.to_datetime(row["quote_date"]).date()
    signal_date = trade_date if trade_date is not None else row_date

    missing = [c for c in feature_cols if c not in row.index]
    if missing:
        raise RuntimeError(f"Feature columns missing from features_spy_latest: {missing}")

    print(f"Running inference: {SYMBOL} | {signal_date}")

    proba = _ensemble_proba(row, feature_cols, fold_models, fold_scalers, fold_weight_map)
    p_down, p_flat, p_up = proba

    context = {
        c: (float(row[c]) if c in row.index and pd.notna(row[c]) else None)
        for c in _CONTEXT_COLS
    }
    signal_dir = _decide_signal(proba, context)

    label = {1: "UP", -1: "DOWN", 0: "FLAT"}[signal_dir]
    print(f"  Signal: {label}  |  P(Up)={p_up:.3f}  P(Flat)={p_flat:.3f}  P(Down)={p_down:.3f}")

    out = {
        "quote_date": str(signal_date),
        "symbol": SYMBOL,
        "signal_dir": signal_dir,
        "proba_up_ens": float(p_up),
        "proba_flat_ens": float(p_flat),
        "proba_down_ens": float(p_down),
        **{c: context.get(c) for c in _CONTEXT_COLS},
    }
    signal_df = pd.DataFrame([out])

    con = duckdb.connect(DUCKDB_PATH_LIVE)
    con.execute(_CREATE_LIVE_SIGNALS)
    con.execute(
        "DELETE FROM live_signals WHERE quote_date = ? AND symbol = ?",
        [str(signal_date), SYMBOL],
    )
    con.execute("INSERT INTO live_signals SELECT * FROM signal_df")
    con.close()

    print(f"  Written to live_signals -> {signal_date} | {SYMBOL} | {label}")


if __name__ == "__main__":
    run_inference()
