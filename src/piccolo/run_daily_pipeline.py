# src/piccolo/run_daily_pipeline.py
"""
Daily pipeline orchestrator — runs all four steps in sequence.

Steps:
  1. Options snapshot      → option_chains table  (DUCKDB_PATH_LIVE_OPTIONS)
  2. EOD prices            → eod_prices table     (DUCKDB_PATH_LIVE)
  3. ML signal generation  → retrain ensemble artifacts, then write today's
                             signal to live_signals (DUCKDB_PATH_LIVE)
  4. Email notification    → signals Excel sent to EMAIL_RECIPIENT

Run manually or via scheduler (cron / Task Scheduler / launchd):

    python -m src.piccolo.run_daily_pipeline

Prerequisites:
  - Theta Data terminal running at 127.0.0.1:25503
  - All .env variables configured (see .env.example)
"""

import subprocess
import sys
from datetime import date, datetime, timedelta


def infer_trade_date() -> date:
    """
    Return today's trade date. If run before 5 AM (e.g. a late overnight run),
    return yesterday to avoid writing a partial day's data.
    """
    now = datetime.now()
    if now.hour < 5:
        return (now - timedelta(days=1)).date()
    return now.date()


def main() -> None:
    python = sys.executable

    print("=" * 60)
    print("  STEP 1: Options Snapshot  (Theta Data)")
    print("=" * 60)
    subprocess.check_call([python, "-m", "src.piccolo.td_options_snapshot"])

    print("\n" + "=" * 60)
    print("  STEP 2: EOD Prices  (Theta Data)")
    print("=" * 60)
    subprocess.check_call([python, "-m", "src.piccolo.eod_prices_td"])

    print("\n" + "=" * 60)
    print("  STEP 3: ML Signal Generation")
    print("=" * 60)
    print("  3a — Retraining ensemble ...")
    subprocess.check_call([python, "-m", "src.piccolo.ml_signal_engine"])
    print("  3b — Running inference ...")
    subprocess.check_call([python, "-m", "src.piccolo.ml_signal_inference"])

    print("\n" + "=" * 60)
    print("  STEP 4: Email Notification")
    print("=" * 60)
    from src.piccolo.notify_email import send
    send(infer_trade_date())


if __name__ == "__main__":
    main()
