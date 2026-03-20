import yfinance as yf
import pandas as pd
import os
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

def collect_stock_history(ticker="AAPL", period="5y"):
    """
    Download and clean historical price data.
    :param ticker: Stock ticker symbol
    :param period: Lookback window (1y, 2y, 5y, max)
    """
    print(f"[Price Collector] Downloading {ticker} price history for {period}...")

    try:
        # 1. Download data
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

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

        # 4. Keep only required columns
        needed_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in needed_cols if c in df.columns]
        df = df[available_cols]

        # 5. Save output
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # Build ticker-specific file name
        save_path = RAW_DIR / f"{ticker}_history.csv"
        
        df.to_csv(save_path, index=False)
        
        print("Price data saved.")
        print(f"Path: {save_path}")
        print(f"Range: {df['Date'].min()} to {df['Date'].max()} ({len(df)} rows)")

    except Exception as e:
        print(f"Price download failed: {e}")

# ==========================================
# 2. Public entry point for main.py
# ==========================================
def run_price_collector(ticker="META"):
    # Use the full available history for backtesting by default.
    collect_stock_history(ticker=ticker, period="max")

if __name__ == "__main__":
    run_price_collector("META")
