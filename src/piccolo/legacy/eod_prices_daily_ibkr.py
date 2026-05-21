# src/piccolo/eod_prices_daily_ibkr.py
"""
Nightly EOD price top-up via the Interactive Brokers API.

For each symbol in LIVE_SYMBOLS, fills any missing daily closes up to
yesterday's session and stores them in the eod_prices table of the live
DuckDB database (DUCKDB_PATH_LIVE).

Schedule this script to run nightly after market close (e.g., via cron).
For initial historical backfill, use bootstrap_eod_prices_ibkr.py instead.

Requirements:
    - IBKR TWS or IB Gateway running and accepting API connections.
    - DUCKDB_PATH_LIVE set in .env (see .env.example).
    - IBKR_HOST and IBKR_PORT set in .env (defaults: 127.0.0.1, 4001).
"""

import os
import threading
import time
from datetime import datetime, date, timedelta

import duckdb
import pandas as pd
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

load_dotenv()

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
CLIENT_ID_START = 10  # different range than bootstrap, to avoid clashes

DUCKDB_PATH_LIVE = os.getenv("DUCKDB_PATH_LIVE")
if not DUCKDB_PATH_LIVE:
    raise ValueError("DUCKDB_PATH_LIVE not set")

from src.piccolo.config_live import LIVE_SYMBOLS


class EODApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.events = {}
        self.connected_event = threading.Event()
        self.client_id_in_use = False

    def nextValidId(self, orderId):
        self.connected_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=None):
        if errorCode == 326:
            self.client_id_in_use = True
            self.connected_event.set()
            return

        if errorCode in (2104, 2106, 2158, 2103, 2119):
            print(f"IB INFO {errorCode} (reqId={reqId}): {errorString}")
            return

        print(f"IB ERROR {errorCode} (reqId={reqId}): {errorString}")
        ev = self.events.get(reqId)
        if ev:
            ev.set()

    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append(bar)

    def historicalDataEnd(self, reqId, start, end):
        ev = self.events.get(reqId)
        if ev:
            ev.set()


def make_contract(symbol: str) -> Contract:
    c = Contract()

    if symbol == "VIX":
        # CBOE VIX index, not a stock
        c.symbol = "VIX"
        c.secType = "IND"
        c.exchange = "CBOE"    # or "CBOEOPT" if that works better in your setup
        c.currency = "USD"
    else:
        # default: stocks/ETFs via SMART
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"

    return c


def connect_eod_app() -> EODApp | None:
    for client_id in range(CLIENT_ID_START, CLIENT_ID_START + 6):
        app = EODApp()
        app.connect(HOST, PORT, clientId=client_id)
        threading.Thread(target=app.run, daemon=True).start()
        print(f"Connecting to {HOST}:{PORT} with clientId={client_id}...")
        app.connected_event.wait(timeout=15)

        if app.client_id_in_use:
            print(f"ClientId {client_id} in use, trying next...")
            app.disconnect()
            time.sleep(1)
            continue

        print(f"Connected with clientId={client_id}")
        return app

    print("Could not connect to IBKR for daily EOD.")
    return None


def fetch_close_for_day(app: EODApp, symbol: str, trade_date: date) -> float | None:
    """
    Get daily bar close for a specific trade_date (historical 1D window).
    """
    end_str = (trade_date + timedelta(days=1)).strftime("%Y%m%d %H:%M:%S")

    req_id = abs(hash((symbol, trade_date))) % 2_000_000_000
    app.events[req_id] = threading.Event()
    app.data[req_id] = []

    contract = make_contract(symbol)

    what_to_show = "TRADES"
    # VIX is an index/vol product; use INDEX (or MIDPOINT) instead of TRADES
    if symbol == "VIX":
        what_to_show = "INDEX"

    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_str,
        durationStr="2 D",
        barSizeSetting="1 day",
        whatToShow=what_to_show,
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[],
    )

    app.events[req_id].wait(timeout=30)
    app.cancelHistoricalData(req_id)

    bars = app.data.get(req_id, [])
    if not bars:
        print(f"{symbol}: no daily bar received for {trade_date}")
        return None

    target_ymd = trade_date.strftime("%Y%m%d")
    for bar in bars:
        d = str(bar.date).split(" ")[0]
        if d == target_ymd:
            print(f"{symbol}: close={bar.close} on {trade_date}")
            return float(bar.close)

    print(f"{symbol}: no matching bar for {trade_date}, bars={len(bars)}")
    return None


def get_live_con(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(DUCKDB_PATH_LIVE), exist_ok=True)
    return duckdb.connect(DUCKDB_PATH_LIVE, read_only=read_only)


def ensure_eod_prices_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS eod_prices (
            symbol     TEXT,
            quote_date DATE,
            close      DOUBLE,
            PRIMARY KEY (symbol, quote_date)
        )
        """
    )


def get_symbol_last_date(con: duckdb.DuckDBPyConnection, symbol: str) -> date | None:
    """
    Return the max quote_date for a symbol in eod_prices, or None if no rows.
    """
    res = con.execute(
        """
        SELECT max(quote_date) AS max_dt
        FROM eod_prices
        WHERE symbol = ?
        """,
        [symbol],
    ).fetchone()
    if not res or res[0] is None:
        return None
    return res[0]


def backfill_prices_until_yesterday() -> None:
    """
    For each LIVE_SYMBOL, fill any missing daily closes up to yesterday.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    con = get_live_con(read_only=False)
    ensure_eod_prices_table(con)
    con.close()

    app = connect_eod_app()
    if app is None:
        return

    try:
        for symbol in LIVE_SYMBOLS:
            print("=" * 60)
            print(f"Backfilling {symbol} EOD prices up to {yesterday}...")

            con = get_live_con(read_only=False)
            ensure_eod_prices_table(con)
            last_dt = get_symbol_last_date(con, symbol)
            con.close()

            # If no history yet, pick a start date (e.g. 3 years back)
            if last_dt is None:
                start_dt = yesterday - timedelta(days=3 * 365)
            else:
                # Start the day after the last stored date
                start_dt = last_dt + timedelta(days=1)

            if start_dt > yesterday:
                print(f"{symbol}: up to date (last={last_dt}).")
                continue

            # Iterate day-by-day (you can later optimize to larger windows)
            cur = start_dt
            rows = []
            while cur <= yesterday:
                # You may want to skip weekends: if cur.weekday() >= 5: cur += 1; continue
                close_px = fetch_close_for_day(app, symbol, cur)
                if close_px is not None:
                    rows.append(
                        {
                            "symbol": symbol,
                            "quote_date": cur,
                            "close": close_px,
                        }
                    )
                cur += timedelta(days=1)
                time.sleep(1)

            if not rows:
                print(f"{symbol}: no new rows fetched.")
                continue

            df = pd.DataFrame(rows)
            df["symbol"] = df["symbol"].astype("object")
            df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.date
            df["close"] = df["close"].astype(float)

            con = get_live_con(read_only=False)
            ensure_eod_prices_table(con)
            con.register("df_new", df)
            con.execute(
                """
                INSERT OR REPLACE INTO eod_prices BY NAME
                SELECT * FROM df_new
                """
            )
            con.close()
            print(f"{symbol}: inserted {len(df)} new rows into eod_prices.")

    finally:
        app.disconnect()
        time.sleep(1)
        print("EOD app disconnected.")


if __name__ == "__main__":
    print(f"Using DUCKDB_PATH_LIVE={DUCKDB_PATH_LIVE}")
    backfill_prices_until_yesterday()
