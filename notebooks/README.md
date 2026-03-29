# Piccolo — Research Notebooks

This directory contains a series of research-quality Jupyter notebooks that
document the full lifecycle of the **Piccolo ML options trading strategy**:
from data ingestion to feature engineering, model training, and backtest
performance analysis.

## How to Run

### Environment Setup

```bash
# From the piccolo-public root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure the following are installed (already in `requirements.txt`):

```
xgboost
duckdb
pandas
numpy
matplotlib
seaborn
scikit-learn
notebook
```

### Kernel

Launch Jupyter from the **piccolo-public root directory** so that the
`src.piccolo` package is importable:

```bash
# From piccolo-public/
jupyter notebook notebooks/
```

Alternatively, add the repo root to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
jupyter notebook
```

## Notebook Order

| # | Notebook | What it covers |
|---|----------|----------------|
| 0 | [00_experiment_plan.ipynb](00_experiment_plan.ipynb) | Research question, hypotheses, methodology overview, pipeline diagram |
| 1 | [01_data_pipeline.ipynb](01_data_pipeline.ipynb) | IBKR + CBOE ingestion, DuckDB schema, data quality checks, sample visualisations |
| 2 | [02_feature_engineering.ipynb](02_feature_engineering.ipynb) | Feature construction, distributions, correlation matrix, label building, class balance |
| 3 | [03_model_training_walkforward.ipynb](03_model_training_walkforward.ipynb) | Walk-forward XGBoost, per-fold metrics, exponential ensemble, confidence threshold tuning |
| 4 | [04_backtest_performance.ipynb](04_backtest_performance.ipynb) | Equity curve vs SPY, Sharpe/drawdown/win rate, monthly heatmap, regime impact, robustness |

Run the notebooks **in order** — each one builds on outputs from the previous.

## Configuration

All hyperparameters are centralised in `src/piccolo/config_strategy.py`.
Do **not** hardcode values in the notebooks — always reference the config
variables (e.g., `cfg.UP_THRESHOLD`).

## Updating Results

After running each notebook:

1. Fill in the **Findings & Notes** table at the bottom of the notebook.
2. Save the notebook with outputs included (do not clear outputs before committing).
3. Update the **Results Summary** in the root `README.md`.
