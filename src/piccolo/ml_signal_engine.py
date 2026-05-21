# src/piccolo/ml_signal_engine.py

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from xgboost import Booster, XGBClassifier
from sklearn.preprocessing import StandardScaler

from config.settings import DUCKDB_PATH_HIST
from src.piccolo.config_strategy import (
    UP_THRESHOLD,
    DOWN_THRESHOLD,
    LABEL_HORIZON_DAYS,
    N_TRAIN_MONTHS,
    N_TEST_MONTHS,
    XGB_PARAMS,
    CONF_THRESHOLD_UP,
    CONF_THRESHOLD_DOWN,
    USE_FLAT_CLASS_FILTER,
    USE_ABOVE_SMA200_FILTER,
    USE_VOL_REGIME_FILTER,
    ALPHA,
)

# Base feature list; filtered per feat_df in build_ml_table
FEATURE_COLS_BASE = [
    "pc_ratio_front", "pc_ratio_back",
    "call_wall_dist_pct", "put_wall_dist_pct", "wall_spread_pct",
    "net_gex", "net_gex_back", "gex_flip",
    "max_pain_dist_pct", "atm_iv", "iv_skew", "oi_concentration",
    "dte_front",
    "sma_200", "above_sma200", "ret_20d", "ret_60d",
    "vol_20d", "daily_ret", "vol_5d", "vol_regime",
]

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_feature_table_spy() -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_PATH_HIST, read_only=True)
    df = con.execute(
        "SELECT * FROM features_spy_latest ORDER BY quote_date"
    ).df()
    con.close()
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df = df.sort_values("quote_date").reset_index(drop=True)
    return df


def build_path_labels(feat_df: pd.DataFrame) -> pd.DataFrame:
    H = LABEL_HORIZON_DAYS
    if H <= 0:
        raise ValueError("LABEL_HORIZON_DAYS must be > 0")

    feat_df = feat_df.copy()

    # 1d forward return (for stats / SPY comp)
    feat_df["fwd_ret_1d"] = feat_df["und_price"].shift(-1) / feat_df["und_price"] - 1

    # Forward multipliers 1..H days ahead
    ret_cols = []
    for k in range(1, H + 1):
        col = f"fwd_mult_{k}d"
        feat_df[col] = feat_df["und_price"].shift(-k) / feat_df["und_price"]
        ret_cols.append(col)

    # Convert to returns and compute path max/min
    fwd_rets = feat_df[ret_cols] - 1.0
    feat_df["path_fwd_ret_max"] = fwd_rets.max(axis=1)
    feat_df["path_fwd_ret_min"] = fwd_rets.min(axis=1)

    # 3-class label based on path extremes
    feat_df["label"] = np.select(
        [
            feat_df["path_fwd_ret_max"] >= UP_THRESHOLD,
            feat_df["path_fwd_ret_min"] <= DOWN_THRESHOLD,
        ],
        [1, -1],
        default=0,
    )

    # Drop rows without full H-day path
    feat_df = feat_df.dropna(
        subset=["path_fwd_ret_max", "path_fwd_ret_min"]
    ).reset_index(drop=True)
    return feat_df


def build_ml_table(feat_df: pd.DataFrame):
    feature_cols_full = [c for c in FEATURE_COLS_BASE if c in feat_df.columns]
    null_pct = feat_df[feature_cols_full].isnull().mean()
    feature_cols = [c for c in feature_cols_full if null_pct[c] < 0.80]

    ml_df = feat_df[["quote_date"] + feature_cols + ["label", "fwd_ret_1d"]].copy()
    for col in feature_cols:
        if ml_df[col].isnull().sum() > 0:
            ml_df[col] = ml_df[col].fillna(ml_df[col].median())

    ml_df = (
        ml_df.dropna(subset=["label"])
        .sort_values("quote_date")
        .reset_index(drop=True)
    )
    ml_df["y"] = ml_df["label"].map(LABEL_MAP)
    return ml_df, feature_cols


def _ensemble_proba_for_row(row, feature_cols, fold_models, fold_scalers, fold_weight_map):
    """
    For a single row of results_df:
    - Take its feature vector x.
    - For EACH fold model:
        - Scale x using that fold's scaler.
        - Predict proba with that fold's model.
    - Combine via fold weights into one probability vector.
    Returns: np.array([p_down, p_flat, p_up])
    """
    x = row[feature_cols].astype(float).values.reshape(1, -1)

    p_acc = np.zeros(3, dtype=float)
    w_acc = 0.0

    for fid, model in fold_models.items():
        scaler = fold_scalers[fid]
        w = fold_weight_map.get(fid, 0.0)
        if w <= 0:
            continue

        x_scaled = scaler.transform(x)
        p = model.predict_proba(x_scaled)[0]  # shape (3,)
        p_acc += w * p
        w_acc += w

    if w_acc > 0:
        return p_acc / w_acc
    else:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)


def train_walkforward(ml_df: pd.DataFrame, feature_cols):
    """
    Walk-forward training with:
    - Per-fold XGBClassifier + StandardScaler.
    - Exponentially weighted ensemble probabilities across folds.
    Returns:
        results_df: with per-fold + ensemble probabilities and preds.
        fold_models: {fold_id: model}
        fold_scalers: {fold_id: scaler}
    """
    ml_df = ml_df.copy()
    ml_df["ym"] = ml_df["quote_date"].dt.to_period("M")
    all_months = sorted(ml_df["ym"].unique())

    print("Total months in data:", len(all_months))
    print("Using N_TRAIN_MONTHS =", N_TRAIN_MONTHS, "N_TEST_MONTHS =", N_TEST_MONTHS)

    results = []
    fold_models = {}
    fold_scalers = {}

    fold_id = 0
    for i in range(N_TRAIN_MONTHS, len(all_months) - N_TEST_MONTHS + 1, N_TEST_MONTHS):
        train_months = all_months[i - N_TRAIN_MONTHS: i]
        test_months = all_months[i: i + N_TEST_MONTHS]

        train = ml_df[ml_df["ym"].isin(train_months)].copy()
        test = ml_df[ml_df["ym"].isin(test_months)].copy()

        if len(train) < 50 or len(test) < 5:
            print(
                f"Skipping fold {fold_id}: "
                f"len(train)={len(train)}, len(test)={len(test)}"
            )
            fold_id += 1
            continue

        X_train = train[feature_cols].values
        y_train = train["y"].values
        X_test = test[feature_cols].values
        y_test = test["y"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = XGBClassifier(
            **XGB_PARAMS,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        fold_models[fold_id] = model
        fold_scalers[fold_id] = scaler

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        test = test.copy()
        test["pred_raw"] = preds
        test["pred_dir"] = test["pred_raw"].map(INV_LABEL_MAP).astype(int)
        test["fold_id"] = fold_id
        test["model_id"] = f"fold_{fold_id}"
        test["proba_down"] = proba[:, 0]
        test["proba_flat"] = proba[:, 1]
        test["proba_up"] = proba[:, 2]
        test["train_start"] = train_months[0].to_timestamp()
        test["train_end"] = train_months[-1].to_timestamp()
        test["test_start"] = test_months[0].to_timestamp()
        test["test_end"] = test_months[-1].to_timestamp()

        results.append(test)
        fold_id += 1

    print("Total folds trained:", len(fold_models))
    print("Total rows in results_df (before concat):", sum(len(r) for r in results))

    if not results:
        raise RuntimeError("No folds were trained; check train/test size thresholds.")

    results_df = pd.concat(results).sort_values("quote_date").reset_index(drop=True)

    # Exponential recency weights over folds (later folds get higher weight)
    all_fold_ids = sorted(fold_models.keys())
    K = len(all_fold_ids)

    raw_weights = np.array([ALPHA ** (K - 1 - idx) for idx in range(K)])
    fold_weights = raw_weights / raw_weights.sum()
    fold_weight_map = {fid: w for fid, w in zip(all_fold_ids, fold_weights)}

    # Apply ensemble to each row
    proba_ens = results_df.apply(
        lambda r: _ensemble_proba_for_row(
            r, feature_cols, fold_models, fold_scalers, fold_weight_map
        ),
        axis=1,
        result_type="expand",
    )
    proba_ens.columns = ["proba_down_ens", "proba_flat_ens", "proba_up_ens"]

    results_df = pd.concat([results_df, proba_ens], axis=1)

    results_df["pred_raw_ens"] = results_df[
        ["proba_down_ens", "proba_flat_ens", "proba_up_ens"]
    ].values.argmax(axis=1)
    results_df["pred_dir_ens"] = results_df["pred_raw_ens"].map(INV_LABEL_MAP).astype(
        int
    )

    return results_df, fold_models, fold_scalers


def _decide_position_row(row):
    # Use ensemble probabilities and ensemble direction
    # Optional: ignore Flat if highest prob
    if USE_FLAT_CLASS_FILTER:
        if (
            row["proba_flat_ens"] >= row["proba_up_ens"]
            and row["proba_flat_ens"] >= row["proba_down_ens"]
        ):
            return 0

    if (
        row["proba_up_ens"] >= CONF_THRESHOLD_UP
        and row["proba_up_ens"] >= row["proba_down_ens"]
    ):
        dir_ = 1
    elif (
        row["proba_down_ens"] >= CONF_THRESHOLD_DOWN
        and row["proba_down_ens"] >= row["proba_up_ens"]
    ):
        dir_ = -1
    else:
        dir_ = 0

    # Regime filters
    if USE_ABOVE_SMA200_FILTER and "above_sma200" in row and row["above_sma200"] != 1:
        dir_ = 0

    if USE_VOL_REGIME_FILTER and "vol_regime" in row and row["vol_regime"] == 0:
        dir_ = 0

    return int(dir_)


def add_signal_columns(results_df: pd.DataFrame):
    """
    Adds:
        signal_dir  : direction from ensemble probabilities + filters.
        position_dir: 1-day lagged signal_dir.
    """
    df = results_df.copy().sort_values("quote_date").reset_index(drop=True)
    df["signal_dir"] = df.apply(_decide_position_row, axis=1).astype(int)
    df["position_dir"] = df["signal_dir"].shift(1).fillna(0).astype(int)
    return df


def export_ensemble(
    artifacts_dir: str,
    results_df: pd.DataFrame,
    fold_models: dict,
    fold_scalers: dict,
    feature_cols: list,
) -> None:
    """
    Save ensemble artifacts to disk so inference can run without retraining.

    - Models: XGBoost JSON files (fold_<id>_model.json)
    - Scalers: numpy arrays (fold_<id>_scaler.npz) storing mean_ and scale_
    - Config: ensemble_config.json with fold_ids, weights, feature_cols, metadata
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Recompute fold weights exactly as in train_walkforward
    all_fold_ids = sorted(fold_models.keys())
    K = len(all_fold_ids)
    raw_weights = np.array([ALPHA ** (K - 1 - idx) for idx in range(K)])
    fold_weights = raw_weights / raw_weights.sum()
    fold_weight_map = {fid: float(w) for fid, w in zip(all_fold_ids, fold_weights)}

    # Save models and scalers per fold
    for fid in all_fold_ids:
        model = fold_models[fid]
        scaler = fold_scalers[fid]

        model_path = artifacts_path / f"fold_{fid}_model.json"
        scaler_path = artifacts_path / f"fold_{fid}_scaler.npz"

        # Save underlying booster instead of sklearn wrapper
        booster = model.get_booster()
        booster.save_model(str(model_path))

        np.savez(
            scaler_path,
            mean_=scaler.mean_,
            scale_=scaler.scale_,
        )

    # Metadata / config
    cfg = {
        "feature_cols": feature_cols,
        "fold_ids": all_fold_ids,
        "fold_weights": fold_weight_map,
        "date_min": str(results_df["quote_date"].min()),
        "date_max": str(results_df["quote_date"].max()),
        "label_horizon_days": LABEL_HORIZON_DAYS,
    }

    cfg_path = artifacts_path / "ensemble_config.json"
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Exported ensemble to {artifacts_path}")


def load_ensemble(artifacts_dir: str):
    """
    Load ensemble artifacts from disk:

    Returns:
      feature_cols: list[str]
      fold_models: dict[fold_id, XGBClassifier]
      fold_scalers: dict[fold_id, StandardScaler-like]
      fold_weight_map: dict[fold_id, float]
    """
    artifacts_path = Path(artifacts_dir)
    cfg_path = artifacts_path / "ensemble_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing ensemble_config.json in {artifacts_path}")

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    feature_cols = cfg["feature_cols"]
    fold_ids = cfg["fold_ids"]
    fw = cfg["fold_weights"]

    # fold_weights may be stored as dict or list; normalize
    if isinstance(fw, dict):
        fold_weight_map = {int(k): float(v) for k, v in fw.items()}
    else:
        fold_weight_map = {int(fid): float(w) for fid, w in zip(fold_ids, fw)}

    fold_models = {}
    fold_scalers = {}

    for fid in fold_ids:
        model_path = artifacts_path / f"fold_{fid}_model.json"
        scaler_path = artifacts_path / f"fold_{fid}_scaler.npz"

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Missing artifacts for fold {fid} in {artifacts_path}")

        # Load booster from JSON
        booster = Booster()
        booster.load_model(str(model_path))

        # Wrap booster in sklearn XGBClassifier and set n_classes_
        model = XGBClassifier(**XGB_PARAMS)
        model._Booster = booster

        n_classes_attr = booster.attr("num_class")
        if n_classes_attr is not None:
            model.n_classes_ = int(n_classes_attr)
        else:
            model.n_classes_ = 3  # 3-class setup: Down / Flat / Up

        data = np.load(scaler_path)
        scaler = StandardScaler()
        scaler.mean_ = data["mean_"]
        scaler.scale_ = data["scale_"]

        fold_models[int(fid)] = model
        fold_scalers[int(fid)] = scaler

    return feature_cols, fold_models, fold_scalers, fold_weight_map


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    ARTIFACTS_DIR = os.getenv(
        "ENSEMBLE_ARTIFACTS_DIR",
        "models/ensemble_spy_latest",  # default if not in .env
    )

    print("Loading SPY feature table...")
    feat_df = load_feature_table_spy()

    print("Building labels...")
    feat_df = build_path_labels(feat_df)

    print("Building ML table...")
    ml_df, feature_cols = build_ml_table(feat_df)

    print("Training walk-forward ensemble...")
    results_df, fold_models, fold_scalers = train_walkforward(ml_df, feature_cols)

    print("Exporting ensemble artifacts to:", ARTIFACTS_DIR)
    export_ensemble(ARTIFACTS_DIR, results_df, fold_models, fold_scalers, feature_cols)
