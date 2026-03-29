# Piccolo — ML Options Strategy Research

**Systematic ML-driven options trading research using options market microstructure signals.**

This repository contains the full research and production pipeline for
**Piccolo**, a machine-learning strategy that uses options market microstructure
signals to predict short-term directional bias in SPY and related ETFs.

---

## Architecture

The high-level data and signal flow is documented in
[`pipelines_flow.mmd`](pipelines_flow.mmd) (Mermaid diagram).  At a glance:

```
IBKR API  ──►  EOD Prices (DuckDB LIVE)   ──►  Feature Engineering
               Options Snapshots (LIVE)   ──►  Walk-Forward XGBoost
                                               ──►  Ensemble Signal
CBOE Historical Data  ──────────────────►          ──►  Backtest / Live
```

---

## Directory Structure

```
piccolo-public/
├── README.md                         ← You are here
├── .env.example                      ← Required environment variables (no real values)
├── .gitignore
├── requirements.txt
├── pipelines_flow.mmd                ← Architecture diagram (Mermaid)
│
├── config/
│   └── settings.py                   ← Env-based config (all paths from .env)
│
├── src/
│   └── piccolo/
│       ├── __init__.py
│       ├── config_strategy.py        ← Strategy hyperparams (example values + docs)
│       ├── config_live.py            ← Live trading symbol universe
│       ├── ml_signal_engine.py       ← Feature loading, labels, walk-forward, ensemble
│       ├── bootstrap_eod_prices_ibkr.py  ← One-time IBKR historical backfill
│       ├── eod_prices_daily_ibkr.py      ← Nightly EOD price top-up
│       └── ibkr_options_snapshot.py      ← Daily options chain capture
│
├── notebooks/
│   ├── README.md
│   ├── 00_experiment_plan.ipynb      ← Hypotheses, methodology, pipeline overview
│   ├── 01_data_pipeline.ipynb        ← Data ingestion and quality checks
│   ├── 02_feature_engineering.ipynb  ← Features, labels, distributions
│   ├── 03_model_training_walkforward.ipynb  ← Walk-forward XGBoost + ensemble
│   └── 04_backtest_performance.ipynb ← Equity curve, Sharpe, drawdown, robustness
│
└── data/
    └── README.md                     ← Data is not included; explains how to obtain
```

---

## Quick Start

### 1. Set Up the Environment

```bash
git clone <repo-url> piccolo-public
cd piccolo-public

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ibapi (IBKR Python API) is not on PyPI — download the installer from:
https://interactivebrokers.github.io/
then: pip install <path-to-ibapi-wheel>

```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your local paths:

```bash
cp .env.example .env
# then edit .env with your DuckDB paths and IBKR connection details
```

### 3. Bootstrap Historical Data

```bash
# One-time IBKR historical price backfill
python src/piccolo/bootstrap_eod_prices_ibkr.py

# Subsequent daily runs (e.g., via cron)
python src/piccolo/eod_prices_daily_ibkr.py
```

### 4. Capture Options Snapshots

```bash
python src/piccolo/ibkr_options_snapshot.py
```

This requires an active IBKR TWS or IB Gateway session.

### 5. Explore the Research Notebooks

```bash
# From the piccolo-public root
jupyter notebook notebooks/
```

Open notebooks in order, starting with `00_experiment_plan.ipynb`.
See [`notebooks/README.md`](notebooks/README.md) for details.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Storage | DuckDB (columnar, file-based) |
| ML Framework | XGBoost (gradient boosted trees) |
| Broker API | IBKR (Interactive Brokers) via `ibapi` |
| Data (historical) | CBOE options data (14 years) |
| Notebooks | Jupyter |
| Visualisation | matplotlib, seaborn |

---

## Key Configuration Parameters

All strategy hyperparameters live in `src/piccolo/config_strategy.py`.
Values are clearly marked as examples — set them to suit your own backtesting.

| Parameter | Description |
|-----------|-------------|
| `UP_THRESHOLD` | Forward return threshold for Up label |
| `DOWN_THRESHOLD` | Forward return threshold for Down label |
| `LABEL_HORIZON_DAYS` | Days ahead for path-forward label construction |
| `N_TRAIN_MONTHS` | Rolling training window size (months) |
| `N_TEST_MONTHS` | Walk-forward test window size (months) |
| `ALPHA` | Exponential ensemble recency weight |
| `CONF_THRESHOLD_UP` | Min ensemble probability to fire an Up signal |
| `CONF_THRESHOLD_DOWN` | Min ensemble probability to fire a Down signal |
| `USE_FLAT_CLASS_FILTER` | Suppress trades when Flat is highest prob class |
| `USE_ABOVE_SMA200_FILTER` | Only go long when price > SMA-200 |
| `USE_VOL_REGIME_FILTER` | Suppress signals in extreme vol regimes |

### Example Usage

```python
import src.piccolo.config_strategy as cfg
from src.piccolo.ml_signal_engine import (
    load_feature_table_spy,
    build_path_labels,
    build_ml_table,
    train_walkforward,
    add_signal_columns,
)

feat_df = load_feature_table_spy()
feat_df = build_path_labels(feat_df)
ml_df, feature_cols = build_ml_table(feat_df)
results_df, fold_models, fold_scalers = train_walkforward(ml_df, feature_cols)
results_df = add_signal_columns(results_df)
```

---

## Results Summary

> **TODO:** Fill in after completing the research notebooks.

| Metric | Value |
|--------|-------|
| Backtest period | _paste_ |
| Annualised return (strategy) | _paste_ |
| Sharpe ratio | _paste_ |
| Max drawdown | _paste_ |
| Win rate | _paste_ |
| SPY B&H return (same period) | _paste_ |

---

## Research Documentation

The full research writeup lives in the notebooks:

- [`notebooks/00_experiment_plan.ipynb`](notebooks/00_experiment_plan.ipynb) — Hypotheses, methodology, pipeline overview
- [`notebooks/01_data_pipeline.ipynb`](notebooks/01_data_pipeline.ipynb) — Data ingestion and quality
- [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) — Features, labels, distributions
- [`notebooks/03_model_training_walkforward.ipynb`](notebooks/03_model_training_walkforward.ipynb) — Model training and ensemble
- [`notebooks/04_backtest_performance.ipynb`](notebooks/04_backtest_performance.ipynb) — Performance analysis and robustness

---

## Disclaimer

This is a research project. Past backtest results do not guarantee future performance.
No trading advice is implied. All threshold values and hyperparameters shown in this
repository are clearly marked as examples and do not represent live trading parameters.
