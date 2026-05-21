# src/piccolo/td_options_snapshot.py
"""
Daily options chain snapshot via Theta Data terminal.

Pulls EOD option chain data (IV, OI, OHLC) for all LIVE_SYMBOLS using the
Theta Data HTTP API (v3). Writes to the `option_chains` table in
DUCKDB_PATH_LIVE_OPTIONS — same schema as the legacy IBKR snapshot.

Data collected per contract:
    - Implied volatility (bid/ask/mid)
    - Open interest
    - OHLC (close, volume)

Requires:
    - Theta Data terminal running at 127.0.0.1:25503
    - DUCKDB_PATH_LIVE_OPTIONS set in .env (see .env.example)
"""

import asyncio
import csv
import io
import os
import time
from datetime import date, datetime, timedelta

import duckdb
import httpx
import pandas as pd

from config.settings import DUCKDB_PATH_LIVE_OPTIONS
from src.piccolo.config_live import LIVE_SYMBOLS

BASE_URL       = 'http://127.0.0.1:25503/v3'
ATM_RANGE      = 0.20   # ±20% around spot — filters out deep OTM contracts
MAX_CONCURRENT = 2      # Theta Data VALUE tier concurrency limit
TARGET_EXPIRIES = 10    # target number of monthly expiries per symbol

RIGHT_MAP = {'CALL': 'C', 'PUT': 'P'}


# ── Expiry discovery ──────────────────────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    """Return the third Friday of a given month — standard monthly expiry date."""
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


def get_monthly_expiries(symbol: str, target_count: int = TARGET_EXPIRIES,
                         max_months: int = 18) -> list[date]:
    """
    Fetch all listed expirations for a symbol from Theta Data.
    Filters to the nearest monthly expiry per calendar month, returns
    target_count expirations going forward.
    """
    raw = []
    with httpx.stream('GET', BASE_URL + '/option/list/expirations',
                      params={'symbol': symbol}, timeout=30) as r:
        if r.status_code != 200:
            print(f'  [{symbol}] expirations HTTP {r.status_code}')
            return []
        header = None
        for line in r.iter_lines():
            for row in csv.reader(io.StringIO(line)):
                if header is None:
                    header = row
                elif len(row) >= 2:
                    raw.append(row[1])

    today = date.today()
    all_exp = sorted(date.fromisoformat(e) for e in raw if e >= str(today))

    result = []
    for i in range(max_months):
        if len(result) >= target_count:
            break
        month = (today.month - 1 + i) % 12 + 1
        year  = today.year + (today.month - 1 + i) // 12
        target = third_friday(year, month)
        candidates = [e for e in all_exp if e.year == year and e.month == month]
        if not candidates:
            continue
        closest = min(candidates, key=lambda e: abs((e - target).days))
        if closest not in result:
            result.append(closest)

    return sorted(result)


# ── Async fetch helpers ───────────────────────────────────────────────────────

async def fetch_stream(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                       endpoint: str, params: dict) -> list[dict]:
    """Generic streaming CSV fetch via Theta Data v3 API. Returns list of row dicts."""
    async with sem:
        rows, header = [], None
        async with client.stream('GET', BASE_URL + endpoint, params=params, timeout=60) as r:
            if r.status_code != 200:
                return []
            async for line in r.aiter_lines():
                for row in csv.reader(io.StringIO(line)):
                    if header is None:
                        header = row
                    else:
                        rows.append(dict(zip(header, row)))
        return rows


async def fetch_iv(client, sem, symbol: str, expiry: date) -> list[dict]:
    return await fetch_stream(client, sem, '/option/snapshot/greeks/implied_volatility',
                              {'symbol': symbol, 'expiration': str(expiry)})


async def fetch_oi(client, sem, symbol: str, expiry: date) -> list[dict]:
    return await fetch_stream(client, sem, '/option/snapshot/open_interest',
                              {'symbol': symbol, 'expiration': str(expiry)})


async def fetch_ohlc(client, sem, symbol: str, expiry: date) -> list[dict]:
    return await fetch_stream(client, sem, '/option/snapshot/ohlc',
                              {'symbol': symbol, 'expiration': str(expiry)})


async def fetch_expiry_data(client, sem, symbol: str, expiry: date) -> list[dict]:
    """
    Fetch IV, OI, and OHLC concurrently for one expiry, then merge into
    unified per-contract rows keyed on (strike, right).
    """
    iv_rows, oi_rows, ohlc_rows = await asyncio.gather(
        fetch_iv(client, sem, symbol, expiry),
        fetch_oi(client, sem, symbol, expiry),
        fetch_ohlc(client, sem, symbol, expiry),
    )

    oi_idx   = {(r['strike'], r['right']): r for r in oi_rows}
    ohlc_idx = {(r['strike'], r['right']): r for r in ohlc_rows}

    merged = []
    for iv in iv_rows:
        key = (iv['strike'], iv['right'])
        merged.append({
            'iv': iv, 'oi': oi_idx.get(key, {}), 'ohlc': ohlc_idx.get(key, {}),
            'strike':    float(iv['strike']),
            'right':     iv['right'],
            'und_price': float(iv['underlying_price']) if iv.get('underlying_price') else None,
        })

    return merged


# ── Row builder ───────────────────────────────────────────────────────────────

def build_row(merged: dict, symbol: str, expiry: date,
              trade_date: date, snapshot_ts: str) -> dict | None:
    """
    Convert a merged IV/OI/OHLC dict into a flat DuckDB row.
    Returns None if bid/ask are missing or mid-price is zero.
    """
    iv, oi, ohlc = merged['iv'], merged['oi'], merged['ohlc']

    bid = float(iv['bid']) if iv.get('bid') else None
    ask = float(iv['ask']) if iv.get('ask') else None
    if bid is None or ask is None:
        return None

    price = round((bid + ask) / 2, 4)
    if price <= 0:
        return None

    return {
        'trade_date':   trade_date,
        'Timestamp':    snapshot_ts,
        'Symbol':       symbol,
        'Expiry':       str(expiry),
        'Strike':       merged['strike'],
        'righttype':    RIGHT_MAP.get(merged['right'], merged['right']),
        'Price':        price,
        'Last':         float(ohlc['close'])  if ohlc.get('close')  else None,
        'Close':        float(ohlc['close'])  if ohlc.get('close')  else None,
        'Bid':          bid,
        'Ask':          ask,
        'Volume':       float(ohlc['volume']) if ohlc.get('volume') else None,
        'OpenInterest': float(oi['open_interest']) if oi.get('open_interest') else None,
        'IV':           float(iv['implied_vol'])   if iv.get('implied_vol')   else None,
        'Delta': None, 'Gamma': None, 'Theta': None, 'Vega': None,
        'UndPrice':     merged['und_price'],
    }


# ── Main snapshot runner ──────────────────────────────────────────────────────

async def run_snapshot(trade_date: date, snapshot_ts: str) -> list[dict]:
    """
    For each symbol in LIVE_SYMBOLS:
      1. Fetch monthly expiries
      2. Fetch IV + OI + OHLC concurrently across expiries
      3. Filter to ATM_RANGE around spot
      4. Build and collect DuckDB rows
    """
    all_results = []
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    t0  = time.time()

    async with httpx.AsyncClient() as client:
        for symbol in LIVE_SYMBOLS:
            print(f"\n{'='*50}\n  {symbol}\n{'='*50}")
            t_sym = time.time()

            expiries = get_monthly_expiries(symbol)
            if not expiries:
                print(f'  No expiries found — skipping.')
                continue
            print(f'  {len(expiries)} expiries: {expiries[0]} … {expiries[-1]}')

            tasks   = [fetch_expiry_data(client, sem, symbol, exp) for exp in expiries]
            results = await asyncio.gather(*tasks)

            sym_rows = 0
            for merged_list in results:
                if not merged_list:
                    continue
                und = next((m['und_price'] for m in merged_list if m['und_price']), None)
                lo  = und * (1 - ATM_RANGE) if und else None
                hi  = und * (1 + ATM_RANGE) if und else None

                for m in merged_list:
                    if lo and not (lo <= m['strike'] <= hi):
                        continue
                    row = build_row(m, symbol, date.fromisoformat(m['iv']['expiration']),
                                    trade_date, snapshot_ts)
                    if row:
                        all_results.append(row)
                        sym_rows += 1

            print(f'  {sym_rows} rows in {time.time()-t_sym:.1f}s')

    print(f"\nTotal: {len(all_results)} rows across all symbols in {time.time()-t0:.1f}s")
    return all_results


# ── DuckDB writer ─────────────────────────────────────────────────────────────

def save_to_duckdb(all_results: list[dict], trade_date: date) -> None:
    if not all_results:
        print('No rows to save.')
        return

    df = pd.DataFrame(all_results)

    expected_cols = [
        'trade_date', 'Timestamp', 'Symbol', 'Expiry', 'Strike', 'righttype',
        'Price', 'Last', 'Close', 'Bid', 'Ask', 'Volume', 'OpenInterest',
        'IV', 'Delta', 'Gamma', 'Theta', 'Vega', 'UndPrice',
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    df = df[expected_cols].copy()

    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
    for col in ['Timestamp', 'Symbol', 'Expiry', 'righttype']:
        df[col] = df[col].astype('object')
    for col in ['Strike', 'Price', 'Last', 'Close', 'Bid', 'Ask',
                'Volume', 'OpenInterest', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega', 'UndPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    os.makedirs(os.path.dirname(DUCKDB_PATH_LIVE_OPTIONS), exist_ok=True)
    con = duckdb.connect(DUCKDB_PATH_LIVE_OPTIONS)
    con.execute("""
        CREATE TABLE IF NOT EXISTS option_chains (
            trade_date DATE, Timestamp TEXT, Symbol TEXT, Expiry TEXT,
            Strike DOUBLE, righttype TEXT, Price DOUBLE, Last DOUBLE,
            Close DOUBLE, Bid DOUBLE, Ask DOUBLE, Volume DOUBLE,
            OpenInterest DOUBLE, IV DOUBLE, Delta DOUBLE, Gamma DOUBLE,
            Theta DOUBLE, Vega DOUBLE, UndPrice DOUBLE
        )
    """)
    con.register('df', df)
    con.execute("""
        INSERT INTO option_chains
        SELECT trade_date, Timestamp, Symbol, Expiry, Strike, righttype,
               Price, Last, Close, Bid, Ask, Volume, OpenInterest,
               IV, Delta, Gamma, Theta, Vega, UndPrice
        FROM df
    """)
    con.close()
    print(f'Saved {len(df)} rows → {DUCKDB_PATH_LIVE_OPTIONS} for {trade_date}')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    now         = datetime.now()
    trade_date  = (now - timedelta(days=1)).date() if now.hour < 7 else now.date()
    snapshot_ts = now.strftime('%Y-%m-%d %H:%M:%S')

    print(f'Snapshot for trade_date={trade_date}  ts={snapshot_ts}')
    results = asyncio.run(run_snapshot(trade_date, snapshot_ts))
    save_to_duckdb(results, trade_date)
