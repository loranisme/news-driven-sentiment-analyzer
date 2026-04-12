import yfinance as yf
import pandas as pd
import os
import requests
from io import StringIO
from datetime import datetime
from pathlib import Path
import sys

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR 

def _symbol_to_file_stem(symbol):
    return symbol.replace("^", "")


def _download_put_call_csv(url, skiprows=2):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), skiprows=skiprows)


def _normalize_put_call_df(df):
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]

    date_col = None
    for c in ['DATE', 'Trade_date', 'date', 'Date']:
        if c in out.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("Put/Call CSV missing date column.")

    ratio_col = None
    for c in ['P/C Ratio', 'Put/Call Ratio', 'P/C', 'Ratio']:
        if c in out.columns:
            ratio_col = c
            break
    if ratio_col is None:
        raise ValueError("Put/Call CSV missing ratio column.")

    call_col = None
    for c in ['CALL', 'CALLS', 'Call', 'Calls']:
        if c in out.columns:
            call_col = c
            break

    put_col = None
    for c in ['PUT', 'PUTS', 'Put', 'Puts']:
        if c in out.columns:
            put_col = c
            break

    total_col = None
    for c in ['TOTAL', 'Total']:
        if c in out.columns:
            total_col = c
            break

    norm = pd.DataFrame()
    norm['date'] = pd.to_datetime(out[date_col], errors='coerce')
    norm['put_call_ratio'] = pd.to_numeric(out[ratio_col], errors='coerce')
    norm['calls'] = pd.to_numeric(out[call_col], errors='coerce') if call_col else pd.NA
    norm['puts'] = pd.to_numeric(out[put_col], errors='coerce') if put_col else pd.NA
    norm['total'] = pd.to_numeric(out[total_col], errors='coerce') if total_col else pd.NA
    norm = norm.dropna(subset=['date', 'put_call_ratio']).copy()
    norm = norm.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    norm['date'] = norm['date'].dt.strftime('%Y-%m-%d')
    return norm


def collect_put_call_history():
    """
    Download CBOE put/call ratio history for total/equity/index.
    """
    print("[Price Collector] Downloading CBOE Put/Call ratio history...")
    os.makedirs(RAW_DIR, exist_ok=True)

    sources = {
        'total': {
            'daily': "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
            'archive': "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv",
        },
        'equity': {
            'daily': "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
            'archive': None,
        },
        'index': {
            'daily': "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv",
            'archive': None,
        },
    }

    for name, cfg in sources.items():
        frames = []
        try:
            daily_df = _download_put_call_csv(cfg['daily'], skiprows=2)
            frames.append(_normalize_put_call_df(daily_df))
        except Exception as exc:
            print(f"   {name} daily download failed: {exc}")

        if cfg['archive']:
            try:
                archive_df = _download_put_call_csv(cfg['archive'], skiprows=2)
                frames.append(_normalize_put_call_df(archive_df))
            except Exception as exc:
                print(f"   {name} archive download failed: {exc}")

        if not frames:
            print(f"   {name}: no data saved.")
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        merged = merged.dropna(subset=['date']).sort_values('date')
        merged = merged.drop_duplicates(subset=['date'], keep='last')
        merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')

        save_path = RAW_DIR / f"CBOE_{name}pc.csv"
        merged.to_csv(save_path, index=False)
        print(f"   Saved {name} put/call: {save_path} ({len(merged)} rows)")

def collect_stock_history(
    ticker="AAPL",
    period="5y",
    auto_adjust=True,
    history_start="2014-01-01",
    history_end=None,
):
    """
    Download and clean historical price data.
    :param ticker: Stock ticker symbol
    :param period: Lookback window (1y, 2y, 5y, max)
    """
    if history_start or history_end:
        print(
            f"[Price Collector] Downloading {ticker} price history "
            f"{history_start or 'start'} -> {history_end or 'latest'} "
            f"(auto_adjust={auto_adjust})..."
        )
    else:
        print(
            f"[Price Collector] Downloading {ticker} price history for {period} "
            f"(auto_adjust={auto_adjust})..."
        )

    try:
        # 1. Download data
        if history_start or history_end:
            df = yf.download(
                ticker,
                start=history_start,
                end=history_end,
                progress=False,
                auto_adjust=auto_adjust,
                actions=True
            )
        else:
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=auto_adjust,
                actions=True
            )

        if df.empty:
            print(f"Error: no data returned for {ticker}.")
            return

        # 2. Normalize MultiIndex columns when needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index
        df = df.reset_index()

        # 3. Format dates
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = df['Date'].dt.tz_localize(None) 
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        split_events = 0
        if 'Stock Splits' in df.columns:
            split_events = int((pd.to_numeric(df['Stock Splits'], errors='coerce').fillna(0.0) != 0.0).sum())

        close_returns = pd.to_numeric(df['Close'], errors='coerce').pct_change(fill_method=None)
        split_like_jumps = int((close_returns.abs() >= 0.55).sum())

        # 4. Keep only required columns
        needed_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in needed_cols if c in df.columns]
        df = df[available_cols]

        # 5. Save output
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # Build ticker-specific file name
        save_path = RAW_DIR / f"{_symbol_to_file_stem(ticker)}_history.csv"
        
        df.to_csv(save_path, index=False)
        
        print("Price data saved.")
        print(f"Path: {save_path}")
        print(f"Range: {df['Date'].min()} to {df['Date'].max()} ({len(df)} rows)")
        print(f"Detected split events from vendor feed: {split_events}")
        index_like_symbol = ticker.startswith("^") or ticker.upper() in {"VIX"}
        if split_like_jumps > 0 and not index_like_symbol:
            print(
                f"Warning: detected {split_like_jumps} split-like daily jumps (>=55%). "
                "Please verify adjustment mode."
            )
        elif split_like_jumps > 0 and index_like_symbol:
            print(
                f"Note: detected {split_like_jumps} large daily jumps in index-like series "
                f"({ticker}); this is usually market-volatility behavior."
            )
        else:
            print("Adjustment sanity check passed: no split-like daily jumps (>=55%).")

    except Exception as e:
        print(f"Price download failed: {e}")

# ==========================================
# 2. Public entry point for main.py
# ==========================================
def run_price_collector(ticker="META"):
    history_start = "2014-01-01"
    collect_stock_history(ticker=ticker, auto_adjust=True, history_start=history_start)
    for aux_symbol in ["SPY", "GOOGL", "QQQ", "SOXX", "SMCI", "^VIX"]:
        collect_stock_history(ticker=aux_symbol, auto_adjust=True, history_start=history_start)
    collect_put_call_history()

if __name__ == "__main__":
    run_price_collector("META")
