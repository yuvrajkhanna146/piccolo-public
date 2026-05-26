# src/piccolo/eod_prices_td.py
"""
Daily EOD close price ingestion via Theta Data terminal.

For each symbol in LIVE_SYMBOLS, fetches any missing daily closes from
(last stored date + 1) up to yesterday. Writes to the `eod_prices` table
in DUCKDB_PATH_LIVE.

Table schema:
    eod_prices (symbol TEXT, quote_date DATE, close DOUBLE)
    Primary key: (symbol, quote_date) — deduplication on re-run.

Endpoints used:
    Equities / ETFs : GET /stock/history/eod
    Index symbols   : GET /index/history/eod  (e.g. VIX)

Requires:
    - Theta Data terminal running at 127.0.0.1:25503
    - DUCKDB_PATH_LIVE set in .env (see .env.example)
"""

import csv
import io
import os
from datetime import date, timedelta

import duckdb
import httpx
import pandas as pd

from config.settings import DUCKDB_PATH_LIVE
from src.piccolo.config_live import LIVE_SYMBOLS

BASE_URL      = 'http://127.0.0.1:25503/v3'
INDEX_SYMBOLS = {'VIX'}   # routed to /index/history/eod instead of /stock/


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_eod_range(symbol: str, start_date: date, end_date: date) -> list[dict]:
    """
    Fetch daily EOD closes for one symbol over [start_date, end_date] inclusive.

    Requests end_date + 5 days extra to ensure the last trading day is captured
    (Theta Data sometimes has a 1-2 day lag), then filters back to the requested
    range before returning.

    Returns list of {'quote_date': date, 'close': float}.
    """
    endpoint  = '/index/history/eod' if symbol in INDEX_SYMBOLS else '/stock/history/eod'
    fetch_end = end_date + timedelta(days=5)

    params = {
        'symbol':     symbol,
        'start_date': start_date.strftime('%Y%m%d'),
        'end_date':   fetch_end.strftime('%Y%m%d'),
    }

    rows, header = [], None
    with httpx.stream('GET', BASE_URL + endpoint, params=params, timeout=30) as r:
        if r.status_code != 200:
            print(f'  [{symbol}] HTTP {r.status_code}')
            return []
        for line in r.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                if header is None:
                    header = row
                else:
                    rows.append(dict(zip(header, row)))

    result = []
    for row in rows:
        created   = row.get('created', '')
        raw_close = row.get('close')
        if not created or not raw_close:
            continue
        try:
            q_date = date.fromisoformat(created[:10])
            if start_date <= q_date <= end_date:
                result.append({'quote_date': q_date, 'close': float(raw_close)})
        except (ValueError, TypeError):
            continue

    return result


# ── DuckDB helpers ────────────────────────────────────────────────────────────

def get_live_con(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(DUCKDB_PATH_LIVE), exist_ok=True)
    return duckdb.connect(DUCKDB_PATH_LIVE, read_only=read_only)


def ensure_eod_prices_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS eod_prices (
            symbol     TEXT,
            quote_date DATE,
            close      DOUBLE,
            PRIMARY KEY (symbol, quote_date)
        )
    """)


def get_last_stored_date(con: duckdb.DuckDBPyConnection, symbol: str) -> date | None:
    """Return the most recent quote_date stored for symbol, or None if no data exists."""
    res = con.execute(
        'SELECT max(quote_date) FROM eod_prices WHERE symbol = ?', [symbol]
    ).fetchone()
    return res[0] if res and res[0] else None


# ── Main backfill ─────────────────────────────────────────────────────────────

def backfill_prices_until_yesterday() -> None:
    """
    For each symbol in LIVE_SYMBOLS:
      - If no history exists: seed with 3 years of history
      - Otherwise: fill from (last_date + 1) to yesterday
    Idempotent — safe to re-run; INSERT OR REPLACE handles duplicates.
    """
    yesterday = date.today() - timedelta(days=1)

    con = get_live_con()
    ensure_eod_prices_table(con)
    con.close()

    for symbol in LIVE_SYMBOLS:
        print('=' * 55)

        con      = get_live_con()
        last_dt  = get_last_stored_date(con, symbol)
        con.close()

        start_dt = (last_dt + timedelta(days=1)) if last_dt else (yesterday - timedelta(days=3 * 365))

        if start_dt > yesterday:
            print(f'{symbol}: up to date (last={last_dt})')
            continue

        rows = fetch_eod_range(symbol, start_dt, yesterday)

        if not rows:
            print(f'{symbol}: no data returned for {start_dt} -> {yesterday}')
            continue

        df = pd.DataFrame(rows)
        df['symbol']     = symbol
        df['quote_date'] = pd.to_datetime(df['quote_date'])
        df['close']      = df['close'].astype(float)
        df = df[['symbol', 'quote_date', 'close']].sort_values('quote_date')

        records = list(zip(
            df['symbol'].tolist(),
            df['quote_date'].dt.strftime('%Y-%m-%d').tolist(),
            df['close'].tolist(),
        ))

        con = get_live_con()
        ensure_eod_prices_table(con)
        con.executemany('INSERT OR REPLACE INTO eod_prices VALUES (?, ?, ?)', records)
        con.close()

        print(f'{symbol}: added {len(df)} rows  '
              f'[{df["quote_date"].min().date()} -> {df["quote_date"].max().date()}]')


if __name__ == '__main__':
    backfill_prices_until_yesterday()
