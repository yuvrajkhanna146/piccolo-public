# src/piccolo/ibkr_options_snapshot.py
"""
Daily options snapshot for LIVE_SYMBOLS.

Logic is matched to the Options/IBGW_OI_Loop pattern:
- Same IBApp callbacks.
- Same ATM discovery via reqMktData on stock.
- Same strike/expiry discovery via reqContractDetails + ATM band.

Differences vs the historical loop:
- Symbols from src.piccolo.config_live.LIVE_SYMBOLS.
- Writes to DUCKDB_PATH_LIVE_OPTIONS.option_chains with trade_date.

Requirements:
    - IBKR TWS or IB Gateway running and accepting API connections.
    - DUCKDB_PATH_LIVE_OPTIONS set in .env (see .env.example).
    - IBKR_HOST and IBKR_PORT set in .env (defaults: 127.0.0.1, 4001).
"""

import os
import sys
import signal
import threading
import time
from datetime import datetime, date, timedelta

import duckdb
import pandas as pd
from dotenv import load_dotenv
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from config.settings import DUCKDB_PATH_LIVE_OPTIONS
from src.piccolo.config_live import LIVE_SYMBOLS

load_dotenv()

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))

MONTHS = 12
ATM_RANGE = 0.5  # 50%
BATCH_SIZE = 90
CANCEL_DRAIN = 2.0
BATCH_WAIT = 15.0

FIELDS = "100,101,105,106,165,225,233,293,294,295,318,411"


# ── Date helpers ──────────────────────────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


def get_monthly_expiries(months_ahead: int = 12):
    expiries, today = [], date.today()
    for i in range(months_ahead):
        month = (today.month - 1 + i) % 12 + 1
        year = today.year + (today.month - 1 + i) // 12
        tf = third_friday(year, month)
        if tf > today:
            expiries.append(tf.strftime("%Y%m%d"))
    return expiries


# ── IB App ────────────────────────────────────────────────────────────────────

class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self._errors = {}
        self._events = {}
        self._lock = threading.Lock()
        self.connected = threading.Event()
        self.client_id_in_use = False

    def nextValidId(self, orderId):
        self.connected.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=None):
        if errorCode in (2104, 2106, 2158, 2103, 2119, 162, 2157):
            return
        if errorCode == 326:
            self.client_id_in_use = True
            self.connected.set()
            return
        if errorCode == 200:
            with self._lock:
                self._errors[reqId] = errorString
            ev = self._events.get(reqId)
            if ev:
                ev.set()
            return
        print(f"  [{errorCode}] reqId={reqId}: {errorString}")

    def _ensure_dict(self, reqId):
        if reqId not in self.data:
            self.data[reqId] = {}

    def tickPrice(self, reqId, tickType, price, attrib):
        names = {
            1: "Bid",
            2: "Ask",
            4: "Last",
            6: "High",
            7: "Low",
            9: "Close",
            14: "Open",
        }
        with self._lock:
            self._ensure_dict(reqId)
            self.data[reqId][names.get(tickType, f"T{tickType}")] = price
        ev = self._events.get(reqId)
        if ev:
            ev.set()

    def tickSize(self, reqId, tickType, size):
        names = {
            8: "Volume",
            27: "CallOpenInterest",
            28: "PutOpenInterest",
            78: "OpenInterest",
        }
        with self._lock:
            self._ensure_dict(reqId)
            self.data[reqId][names.get(tickType, f"T{tickType}")] = size
        ev = self._events.get(reqId)
        if ev:
            ev.set()

    def tickOptionComputation(
        self,
        reqId,
        tickType,
        tickAttrib,
        impliedVol,
        delta,
        optPrice,
        pvDividend,
        gamma,
        vega,
        theta,
        undPrice,
    ):
        prefix = {10: "Bid", 11: "Ask", 12: "Last", 13: "Model"}.get(
            tickType, f"T{tickType}"
        )
        with self._lock:
            self._ensure_dict(reqId)
            self.data[reqId].update(
                {
                    f"{prefix}_IV": impliedVol,
                    f"{prefix}_Delta": delta,
                    f"{prefix}_Gamma": gamma,
                    f"{prefix}_Theta": theta,
                    f"{prefix}_Vega": vega,
                    f"{prefix}_OptPrice": optPrice,
                    f"{prefix}_UndPrice": undPrice,
                }
            )
        ev = self._events.get(reqId)
        if ev:
            ev.set()

    def contractDetails(self, reqId, contractDetails):
        with self._lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        ev = self._events.get(reqId)
        if ev:
            ev.set()


# ── Contract helpers ──────────────────────────────────────────────────────────

def make_stk(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def make_opt(symbol: str, expiry: str, strike: float, right: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "OPT"
    c.exchange = "SMART"
    c.currency = "USD"
    c.multiplier = "100"
    c.lastTradeDateOrContractMonth = expiry
    c.strike = float(strike)
    c.right = right
    return c


def make_chain_req(symbol: str, expiry: str, right: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "OPT"
    c.exchange = "SMART"
    c.currency = "USD"
    c.lastTradeDateOrContractMonth = expiry
    c.right = right
    return c


def req_contract_details(app: IBApp, req_id: int, contract: Contract, timeout: float = 20.0):
    ev = threading.Event()
    app._events[req_id] = ev
    app.data[req_id] = []
    app.reqContractDetails(req_id, contract)
    ev.wait(timeout=timeout)
    return app.data.get(req_id, [])


# ── ATM price helper ──────────────────────────────────────────────────────────

def get_atm_price(app: IBApp, req_id: int, symbol: str, retries: int = 3):
    for attempt in range(retries):
        app.data[req_id] = {}
        app._events[req_id] = threading.Event()
        app.reqMktData(req_id, make_stk(symbol), "", False, False, [])
        time.sleep(5.0)
        app.cancelMktData(req_id)
        time.sleep(0.5)

        d = app.data.get(req_id, {})
        price = d.get("Last") or d.get("Close") or d.get("Bid") or d.get("Ask")
        if price and price > 0:
            return price

        print(f"  [{symbol}] No price on attempt {attempt+1}/{retries}, retrying...")
        time.sleep(5.0)

    return None


# ── Row extraction ─────────────────────────────────────────────────────────────

def extract_row(symbol: str, expiry: str, strike: float, right: str, mkt: dict, trade_date: date):
    price = next(
        (mkt[k] for k in ("Last", "Close", "Model_OptPrice") if mkt.get(k) and mkt[k] > 0),
        None,
    )
    if price is None:
        return None

    oi = mkt.get("CallOpenInterest") if right == "C" else mkt.get("PutOpenInterest")

    return {
        "trade_date": trade_date,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Expiry": expiry,
        "Strike": float(strike),
        "righttype": right,
        "Price": float(price),
        "Last": mkt.get("Last"),
        "Close": mkt.get("Close"),
        "Bid": mkt.get("Bid"),
        "Ask": mkt.get("Ask"),
        "Volume": mkt.get("Volume"),
        "OpenInterest": oi,
        "IV": mkt.get("Model_IV"),
        "Delta": mkt.get("Model_Delta"),
        "Gamma": mkt.get("Model_Gamma"),
        "Theta": mkt.get("Model_Theta"),
        "Vega": mkt.get("Model_Vega"),
        "UndPrice": mkt.get("Model_UndPrice"),
    }


# ── Signal handler ────────────────────────────────────────────────────────────

_app = None


def _bye(sig, frame):
    print("\nCtrl+C — disconnecting...")
    if _app:
        try:
            _app.disconnect()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, _bye)


# ── Main snapshot logic ────────────────────────────────────────────────────────

def run_snapshot(trade_date: date):
    global _app

    # Auto-retry client IDs to avoid "already in use" error
    app = None
    for client_id in range(4, 10):
        _candidate = IBApp()
        _candidate.connect(HOST, PORT, client_id)
        threading.Thread(target=_candidate.run, daemon=True).start()
        print(f"Connecting to {HOST}:{PORT} (client_id={client_id})...")
        _candidate.connected.wait(timeout=15)
        if not _candidate.client_id_in_use:
            app = _candidate
            _app = app
            print(f"Connected with client ID {client_id}")
            break
        print(f"  Client ID {client_id} in use, trying next...")
        _candidate.disconnect()
        time.sleep(1.0)

    if app is None:
        print("Could not find a free client ID. Wait 30s and retry.")
        return

    try:
        app.reqMarketDataType(3)  # 3 = Delayed, 4 = Delayed Frozen
        expiries = get_monthly_expiries(MONTHS)
        all_results = []
        req_id = 1000

        for symbol in LIVE_SYMBOLS:
            print(f"\n{'=' * 50}\n  {symbol}\n{'=' * 50}")

            # Step 1 — get ATM price
            atm = get_atm_price(app, req_id, symbol)
            req_id += 1
            if not atm or atm <= 0:
                print(f"  Could not get price for {symbol}, skipping.")
                continue
            lo, hi = atm * (1 - ATM_RANGE), atm * (1 + ATM_RANGE)
            print(f"  ATM: ${atm:.2f}  |  Strikes: ${lo:.2f} to ${hi:.2f}")

            # Step 2 — discover filtered strikes (via contractDetails)
            contracts_to_fetch = []
            for expiry in expiries:
                for right in ("C", "P"):
                    details = req_contract_details(app, req_id, make_chain_req(symbol, expiry, right))
                    req_id += 1
                    all_s = sorted(set(cd.contract.strike for cd in details))
                    for s in all_s:
                        if lo <= s <= hi:
                            contracts_to_fetch.append((expiry, s, right))
                    time.sleep(0.3)

            total = len(contracts_to_fetch)
            n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE if total > 0 else 0
            est_secs = n_batches * (BATCH_WAIT + 1)
            print(f"  {total} contracts | {n_batches} batches | ~{est_secs}s estimated")

            if total == 0:
                continue

            # Step 3 — concurrent batch fetching
            for b_start in range(0, total, BATCH_SIZE):
                batch = contracts_to_fetch[b_start : b_start + BATCH_SIZE]
                batch_ids = []

                for expiry, strike, right in batch:
                    app._events[req_id] = threading.Event()
                    app.reqMktData(
                        req_id,
                        make_opt(symbol, expiry, strike, right),
                        FIELDS,
                        False,
                        False,
                        [],
                    )
                    batch_ids.append((req_id, expiry, strike, right))
                    req_id += 1

                time.sleep(BATCH_WAIT)

                for bid, expiry, strike, right in batch_ids:
                    app.cancelMktData(bid)
                    time.sleep(0.05)

                for bid, expiry, strike, right in batch_ids:
                    if not app._errors.get(bid):
                        mkt = app.data.get(bid, {})
                        if mkt:
                            row = extract_row(symbol, expiry, strike, right, mkt, trade_date)
                            if row:
                                all_results.append(row)

                time.sleep(CANCEL_DRAIN)
                done = min(b_start + BATCH_SIZE, total)
                print(f"  {done}/{total} done...")
                time.sleep(1.0)

        if not all_results:
            print("No option rows collected.")
            return

        # Step 4 — save to DuckDB: append with trade_date
        df = pd.DataFrame(all_results)

        expected_cols = [
            "trade_date", "Timestamp", "Symbol", "Expiry", "Strike", "righttype",
            "Price", "Last", "Close", "Bid", "Ask", "Volume", "OpenInterest",
            "IV", "Delta", "Gamma", "Theta", "Vega", "UndPrice",
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df = df[expected_cols].copy()

        # dates as Python date objects
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        # cast to standard Python string (object dtype), NOT pandas "string" dtype
        for col in ["Timestamp", "Symbol", "Expiry", "righttype"]:
            df[col] = df[col].astype("object")

        numeric_cols = [
            "Strike", "Price", "Last", "Close", "Bid", "Ask",
            "Volume", "OpenInterest", "IV", "Delta", "Gamma",
            "Theta", "Vega", "UndPrice",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print("DF dtypes before DuckDB:\n", df.dtypes)

        con = duckdb.connect(DUCKDB_PATH_LIVE_OPTIONS)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS option_chains (
                trade_date DATE,
                Timestamp TEXT,
                Symbol TEXT,
                Expiry TEXT,
                Strike DOUBLE,
                righttype TEXT,
                Price DOUBLE,
                Last DOUBLE,
                Close DOUBLE,
                Bid DOUBLE,
                Ask DOUBLE,
                Volume DOUBLE,
                OpenInterest DOUBLE,
                IV DOUBLE,
                Delta DOUBLE,
                Gamma DOUBLE,
                Theta DOUBLE,
                Vega DOUBLE,
                UndPrice DOUBLE
            )
            """
        )
        con.register("df", df)
        con.execute(
            """
            INSERT INTO option_chains
            SELECT trade_date, Timestamp, Symbol, Expiry, Strike, righttype,
                   Price, Last, Close, Bid, Ask, Volume, OpenInterest,
                   IV, Delta, Gamma, Theta, Vega, UndPrice
            FROM df
            """
        )
        con.close()
        print(f"\nSaved {len(df)} rows -> {DUCKDB_PATH_LIVE_OPTIONS}.option_chains for {trade_date}")

    finally:
        app.disconnect()
        time.sleep(1.0)
        print("Done!")


if __name__ == "__main__":
    now = datetime.now()
    # EOD heuristic: runs before 7am are treated as prior trade day
    if now.hour < 7:
        trade_date = (now - timedelta(days=1)).date()
    else:
        trade_date = now.date()

    print(f"Writing snapshot for trade_date={trade_date} to {DUCKDB_PATH_LIVE_OPTIONS}")
    run_snapshot(trade_date=trade_date)
