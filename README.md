# Piccolo — ML Options Strategy Research

**Systematic ML-driven options trading research using options market microstructure signals.**

This repository contains the full research and production pipeline for
**Piccolo**, a machine-learning strategy that uses 13 options market microstructure
features (GEX, IV skew, OI concentration, put/call ratios, max pain distance,
ATM IV, DTE) to predict short-term directional bias in SPY and related ETFs.

---

## Architecture

The high-level data and signal flow is documented in
[`pipelines_flow.mmd`](pipelines_flow.mmd) (Mermaid diagram).  At a glance:

```
Theta Data  ──►  EOD Prices (DuckDB LIVE)   ──►  Feature Engineering
                 Options Snapshots (LIVE)   ──►  Walk-Forward XGBoost
                                                 ──►  Ensemble Signal  ──►  Email
CBOE Historical Data  ──────────────────►              ──►  Backtest
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
│       ├── config_strategy.py            ← Strategy hyperparams (example values + docs)
│       ├── config_live.py                ← Live trading symbol universe
│       ├── ml_signal_engine.py           ← Feature loading, labels, walk-forward, ensemble
│       ├── ml_signal_inference.py        ← Load saved ensemble → write signal to live_signals
│       ├── run_daily_pipeline.py         ← Daily pipeline orchestrator (Steps 1–4)
│       ├── td_options_snapshot.py        ← Theta Data: daily options chain ingestion
│       ├── eod_prices_td.py              ← Theta Data: EOD price ingestion
│       ├── notify_email.py               ← Signal email notification (Gmail)
│       └── legacy/                       ← IBKR-based scripts (superseded by Theta Data)
│           ├── bootstrap_eod_prices_ibkr.py  ← One-time historical backfill
│           ├── eod_prices_daily_ibkr.py      ← Nightly EOD price top-up
│           └── ibkr_options_snapshot.py      ← Daily options chain capture
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
```

> **Legacy IBKR users only:** `ibapi` is not on PyPI. Download from
> https://interactivebrokers.github.io/ and install the wheel manually.
> Not required for the Theta Data pipeline.

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your local paths:

```bash
cp .env.example .env
# then edit .env with your DuckDB paths and connection details
```

### 3. Bootstrap Historical Data (IBKR users only)

> **Theta Data users can skip this step** — `eod_prices_td.py` seeds 3 years of
> price history automatically on first run.

```bash
# One-time historical price backfill via IBKR
python src/piccolo/legacy/bootstrap_eod_prices_ibkr.py
```

### 4. Run the Daily Pipeline (Theta Data)

With the Theta Data terminal running at `127.0.0.1:25503`, the full pipeline
(options snapshot → EOD prices → model retraining → email notification) runs as:

```bash
python -m src.piccolo.run_daily_pipeline
```

Or run each step individually:

```bash
python -m src.piccolo.td_options_snapshot    # Step 1: options chain
python -m src.piccolo.eod_prices_td          # Step 2: EOD prices
python -m src.piccolo.ml_signal_engine       # Step 3a: model retraining
python -m src.piccolo.ml_signal_inference    # Step 3b: inference → live_signals
```

**Legacy IBKR data scripts** (still functional, now superseded by Theta Data):

```bash
python src/piccolo/legacy/bootstrap_eod_prices_ibkr.py
python src/piccolo/legacy/eod_prices_daily_ibkr.py
python src/piccolo/legacy/ibkr_options_snapshot.py
```

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
| Market data (live) | Theta Data terminal (options chain + EOD prices) |
| Data ingestion (legacy) | IBKR (Interactive Brokers) via `ibapi` — superseded by Theta Data |
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

### Results Summary

| Metric | Strategy | SPY B&H |
|--------|----------|---------|
| Backtest period | Feb 2015 – Feb 2026 | |
| Total return | 3,138.6% | 241.4% |
| Annualised return | 39.3% | 12.4% |
| Sharpe ratio | 2.172 | 0.762 |
| Max drawdown | −31.5% | −34.7% |
| Win rate | 57.6% | 55.4% |
| Profit factor | 1.707 | 1.157 |
| Days in market | 58.4% | 100% |

**Sub-period performance:**

| Period | Ann. Return | Sharpe | Max DD |
|--------|-------------|--------|--------|
| Pre-COVID (before 2020) | −2.1% | −0.120 | −25.7% |
| COVID Crisis (2020) | 86.8% | 2.271 | −14.5% |
| Recovery (2021–2022) | 35.4% | 2.057 | −9.1% |
| Bear Market (2022) | 211.7% | 5.014 | −9.1% |
| Bull (2023–present) | 80.1% | 4.541 | −7.2% |

> **Note:** Results are from a frictionless backtest (no transaction costs or slippage).
> The strategy's performance is concentrated in the post-2020 regime; pre-2020
> returns were negative. See [04_backtest_performance.ipynb](notebooks/04_backtest_performance.ipynb)
> for full analysis, robustness checks, and limitations.

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
