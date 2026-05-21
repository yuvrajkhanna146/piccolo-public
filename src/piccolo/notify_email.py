# src/piccolo/notify_email.py
"""
Post-pipeline email notification with Excel signal report.

Reads today's signals from the live_signals table, builds an Excel workbook,
saves it locally to the signals/ output directory, then emails it as an
attachment via Gmail SMTP.

Required .env vars:
    EMAIL_SENDER        Sending Gmail address
    EMAIL_RECIPIENT     Recipient address
    EMAIL_SMTP_PASSWORD Gmail App Password (16 chars, no spaces)
                        Generate at: myaccount.google.com/apppasswords
"""

import io
import os
import smtplib
from datetime import date
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import duckdb
import pandas as pd
from dotenv import load_dotenv

from config.settings import DUCKDB_PATH_LIVE

load_dotenv()

SENDER    = os.getenv('EMAIL_SENDER')
RECIPIENT = os.getenv('EMAIL_RECIPIENT')
SMTP_PASS = os.getenv('EMAIL_SMTP_PASSWORD')
SMTP_HOST = 'smtp.gmail.com'
SMTP_PORT = 587

# Local directory for saving signal Excel files
SIGNALS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'signals')

SIGNAL_LABELS = {1: 'BUY', -1: 'SELL', 0: 'FLAT'}


# ── Data loader ───────────────────────────────────────────────────────────────

def load_signals(trade_date: date) -> pd.DataFrame:
    """Load all live_signals rows for the given trade_date."""
    con = duckdb.connect(DUCKDB_PATH_LIVE, read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM live_signals WHERE quote_date = ? ORDER BY symbol",
            [trade_date],
        ).df()
    except Exception as e:
        if 'live_signals' in str(e):
            raise RuntimeError(
                'live_signals table not found — has ml_signal_inference.py been run?'
            ) from e
        raise
    finally:
        con.close()
    return df


# ── Excel builder ─────────────────────────────────────────────────────────────

def build_excel(df: pd.DataFrame) -> bytes:
    """
    Format the signals DataFrame into an Excel workbook.
    Applies human-readable column names and auto-sizes columns.
    """
    out = df.copy()
    out['signal_dir'] = out['signal_dir'].map(SIGNAL_LABELS).fillna('FLAT')

    for col in ['proba_up_ens', 'proba_flat_ens', 'proba_down_ens',
                'ret_20d', 'ret_60d', 'vol_20d', 'vol_5d', 'daily_ret']:
        if col in out.columns:
            out[col] = out[col].round(4)

    out = out.rename(columns={
        'quote_date':      'Date',
        'symbol':          'Symbol',
        'signal_dir':      'Signal',
        'proba_up_ens':    'P(Up)',
        'proba_flat_ens':  'P(Flat)',
        'proba_down_ens':  'P(Down)',
        'above_sma200':    'Above SMA200',
        'vol_regime':      'Vol Regime',
        'ret_20d':         'Ret 20D',
        'ret_60d':         'Ret 60D',
        'vol_20d':         'Vol 20D',
        'vol_5d':          'Vol 5D',
        'daily_ret':       'Daily Ret',
        'sma_200':         'SMA 200',
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        out.to_excel(writer, index=False, sheet_name='Signals')
        ws = writer.sheets['Signals']
        for col in ws.columns:
            width = max(len(str(cell.value or '')) for cell in col) + 3
            ws.column_dimensions[col[0].column_letter].width = width
    return buf.getvalue()


# ── Send ──────────────────────────────────────────────────────────────────────

def send(trade_date: date) -> None:
    """
    Load signals for trade_date, save Excel locally, and email to RECIPIENT.
    Raises ValueError if any required env var is missing.
    Skips silently if no signals exist for the given date.
    """
    if not all([SENDER, RECIPIENT, SMTP_PASS]):
        raise ValueError(
            'EMAIL_SENDER, EMAIL_RECIPIENT, EMAIL_SMTP_PASSWORD must be set in .env'
        )

    df = load_signals(trade_date)
    if df.empty:
        print(f'No signals for {trade_date} — skipping email.')
        return

    buys  = int((df['signal_dir'] ==  1).sum())
    sells = int((df['signal_dir'] == -1).sum())
    flats = int((df['signal_dir'] ==  0).sum())

    excel = build_excel(df)

    # Save locally
    os.makedirs(SIGNALS_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(SIGNALS_OUTPUT_DIR, f'piccolo_signals_{trade_date}.xlsx')
    with open(save_path, 'wb') as f:
        f.write(excel)
    print(f'Signals saved → {save_path}')

    # Build email
    msg = MIMEMultipart()
    msg['From']    = SENDER
    msg['To']      = RECIPIENT
    msg['Subject'] = f'Piccolo Signals — {trade_date}'
    msg.attach(MIMEText(
        f'Pipeline completed for {trade_date}.\n\n'
        f'  BUY:  {buys}\n'
        f'  SELL: {sells}\n'
        f'  FLAT: {flats}\n\n'
        f'Full signal table attached.',
        'plain'
    ))

    att = MIMEApplication(excel, _subtype='xlsx')
    att.add_header('Content-Disposition', 'attachment',
                   filename=f'piccolo_signals_{trade_date}.xlsx')
    msg.attach(att)

    # Send
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(SENDER, SMTP_PASS)
        smtp.sendmail(SENDER, RECIPIENT, msg.as_string())

    print(f'Email sent → {RECIPIENT}')


if __name__ == '__main__':
    send(date.today())
