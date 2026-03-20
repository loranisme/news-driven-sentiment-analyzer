import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR

# ==========================================
# 2. Technical forecast model
# ==========================================
class TechForecaster:
    def __init__(self, ticker="TSLA"):
        self.ticker = ticker
        self.file_path = RAW_DIR / f"{ticker}_history.csv"
        
    def add_indicators(self, df):
        df = df.copy()
        # Moving averages
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Momentum
        df['Momentum'] = df['close'] - df['close'].shift(10)
        
        # Volatility
        df['Volatility'] = df['close'].rolling(window=10).std()
        
        return df.dropna()

    def run_forecast(self):
        print(f"\n{'=' * 56}")
        print(f" Technical Forecast | {self.ticker}")
        print(f"{'=' * 56}")
        
        if not os.path.exists(self.file_path):
            print(f"Error: data file not found: {self.file_path}")
            print("Run price collection first.")
            return
            
        df = pd.read_csv(self.file_path)

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Validate required columns
        if 'date' not in df.columns:
            print("Error: data format is invalid. Missing 'date' column.")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        print(f"Latest market date: {df.index[-1].date()}")
        
        # Build features
        df_featured = self.add_indicators(df)
        
        if len(df_featured) < 50:
            print("Error: not enough data for technical forecasting.")
            return

        # Target: next session up (1) or down (0)
        df_featured['next_close'] = df_featured['close'].shift(-1)
        df_featured['Target'] = (df_featured['next_close'] > df_featured['close']).astype(float)
        
        features = ['SMA_10', 'SMA_50', 'RSI', 'Momentum', 'Volatility']
        
        # Use the latest row as the next-session input
        last_row = df_featured.iloc[[-1]][features]
        
        # Exclude the last row because its future label is unknown
        train_data = df_featured.iloc[:-1].dropna(subset=['Target']).copy()
        train_data['Target'] = train_data['Target'].astype(int)
        
        X = train_data[features]
        y = train_data['Target']
        
        # Train model
        print("Training technical classifier on historical data...")
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        # Forecast
        print("Generating next-session probability estimate...")
        probability = model.predict_proba(last_row)[0]
        
        print("\n" + "-" * 56)
        print(f" Forecast Summary | {self.ticker}")
        print("-" * 56)
        
        # probability[1] is the probability of an upward move
        up_prob = probability[1] * 100
        
        if up_prob > 55:
            print("Direction : Bullish")
            print(f"Confidence: {up_prob:.2f}%")
            print("View      : Technical signals are broadly supportive.")
        elif up_prob < 45:
            print("Direction : Bearish")
            print(f"Confidence: {(100-up_prob):.2f}%")
            print("View      : Momentum is weakening or risk is elevated.")
        else:
            print("Direction : Neutral")
            print(f"Confidence: {up_prob:.2f}%")
            print("View      : Signals are mixed and lack clear direction.")
            
        print("-" * 56 + "\n")
        
        curr_rsi = last_row['RSI'].values[0]
        curr_price = df.iloc[-1]['close']
        sma_50 = last_row['SMA_50'].values[0]
        
        print("Reference indicators:")
        print(f"  Close price : ${curr_price:.2f}")
        print(f"  RSI (14)    : {curr_rsi:.2f}")
        print(f"  SMA 50      : ${sma_50:.2f}")
        
        if curr_price < sma_50:
            print("Trend note   : Price is below the 50-day average. Long-term trend is weaker.")
        elif curr_price > sma_50:
             print("Trend note   : Price is above the 50-day average. Long-term trend is constructive.")

# ==========================================
# 3. Public entry point for main.py
# ==========================================
def run_tech_forecast(ticker="META"):
    forecaster = TechForecaster(ticker=ticker)
    forecaster.run_forecast()

if __name__ == "__main__":
    run_tech_forecast("META")
