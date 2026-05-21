# src/piccolo/bootstrap_eod_prices_ibkr.py
"""
One-time historical EOD price backfill via the Interactive Brokers API.

Fetches daily OHLCV bars for SYMBOLS over HIST_DURATION and stores them
in the eod_prices table of the live DuckDB database (DUCKDB_PATH_LIVE).

Run this once to bootstrap price history before starting nightly top-ups
with eod_prices_daily_ibkr.py.

Requirements:
    - IBKR TWS or IB Gateway running and accepting API connections.
    - DUCKDB_PATH_LIVE set in .env (see .env.example).
    - IBKR_HOST and IBKR_PORT set in .env (defaults: 127.0.0.1, 4001).
"""

import os
import threading
import time
from datetime import datetime, date

import duckdb
import pandas as pd
from dotenv import load_dotenv

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

load_dotenv()

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
CLIENT_ID_START = 4

SYMBOLS = ["SPY", "QQQ", "VOO", "AAPL"]

HIST_DURATION = "2 Y"
BAR_SIZE = "1 day"
WHAT_TO_SHOW = "TRADES"
USE_RTH = 1
FORMAT_DATE = 1

DUCKDB_PATH_LIVE = os.getenv("DUCKDB_PATH_LIVE")
if not DUCKDB_PATH_LIVE:
    raise ValueError("DUCKDB_PATH_LIVE is not set in .env")


class HistApp(EWrapper, EClient):
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

        # Ignore common info/warning codes
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
        self.data[reqId].append(
            {
                "time": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )

    def historicalDataEnd(self, reqId, start, end):
        ev = self.events.get(reqId)
        if ev:
            ev.set()


def make_stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def connect_hist_app() -> HistApp | None:
    for client_id in range(CLIENT_ID_START, CLIENT_ID_START + 6):
        app = HistApp()
        app.connect(HOST, PORT, clientId=client_id)
        threading.Thread(target=app.run, daemon=True).start()

        print(f"Connecting to {HOST}:{PORT} with clientId={client_id}...")
        app.connected_event.wait(timeout=15)

        if app.client_id_in_use:
            print(f"ClientId {client_id} is already in use, trying next...")
            app.disconnect()
            time.sleep(1)
            continue

        print(f"Connected with clientId={client_id}")
        return app

    print("Could not connect to IBKR (all client IDs in range are in use).")
    return None


def fetch_history_for_symbol(app: HistApp, symbol: str) -> pd.DataFrame:
    # Empty string = "now" in instrument timezone (avoids timezone warning).
    end_str = ""

    req_id = abs(hash(symbol)) % 2_000_000_000
    app.events[req_id] = threading.Event()
    app.data[req_id] = []

    contract = make_stock_contract(symbol)

    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_str,
        durationStr=HIST_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=USE_RTH,
        formatDate=FORMAT_DATE,
        keepUpToDate=False,
        chartOptions=[],
    )

    app.events[req_id].wait(timeout=60)
    app.cancelHistoricalData(req_id)

    bars = app.data.get(req_id, [])
    if not bars:
        print(f"{symbol}: no historical bars received.")
        return pd.DataFrame(columns=["quote_date", "close"])

    df = pd.DataFrame(bars)

    def to_date(s):
        s = str(s)
        if " " in s:
            s = s.split(" ")[0]
        return datetime.strptime(s, "%Y%m%d").date()

    df["quote_date"] = df["time"].apply(to_date)
    df["close"] = df["close"].astype(float)

    df = df[["quote_date", "close"]].drop_duplicates("quote_date")
    df = df.sort_values("quote_date").reset_index(drop=True)

    print(f"{symbol}: fetched {len(df)} daily bars.")
    return df


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


def upsert_eod_prices(df_all: pd.DataFrame) -> None:
    if df_all.empty:
        print("No data to insert, skipping DuckDB upsert.")
        return

    # Force simple dtypes to avoid DuckDB 'str' dtype issue with newer pandas.
    df_all = df_all.copy()
    df_all["symbol"] = df_all["symbol"].astype("object")
    df_all["quote_date"] = pd.to_datetime(df_all["quote_date"]).dt.date
    df_all["close"] = df_all["close"].astype(float)

    con = get_live_con(read_only=False)
    ensure_eod_prices_table(con)

    symbols = df_all["symbol"].unique().tolist()
    min_date = df_all["quote_date"].min()
    max_date = df_all["quote_date"].max()

    con.execute(
        """
        DELETE FROM eod_prices
        WHERE quote_date BETWEEN ? AND ?
          AND symbol IN (SELECT UNNEST(?)::TEXT)
        """,
        [min_date, max_date, symbols],
    )

    # Register df_all as a DuckDB relation and insert BY NAME.
    con.register("df_all", df_all)
    con.execute("INSERT INTO eod_prices BY NAME SELECT * FROM df_all")
    con.close()
    print("Upsert into eod_prices completed.")


def bootstrap_eod_prices() -> None:
    app = connect_hist_app()
    if app is None:
        return

    all_rows = []

    try:
        for symbol in SYMBOLS:
            print("=" * 50)
            print(f"Fetching history for {symbol} ...")
            df = fetch_history_for_symbol(app, symbol)
            if df.empty:
                continue
            df["symbol"] = symbol
            all_rows.append(df[["symbol", "quote_date", "close"]])
            time.sleep(2)
    finally:
        app.disconnect()
        time.sleep(1)

    if not all_rows:
        print("No historical data fetched for any symbol.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    print(f"Total rows to upsert: {len(df_all)}")
    upsert_eod_prices(df_all)


if __name__ == "__main__":
    print(f"Using DUCKDB_PATH_LIVE={DUCKDB_PATH_LIVE}")
    bootstrap_eod_prices()
