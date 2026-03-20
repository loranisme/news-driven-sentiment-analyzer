import os
import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from sklearn.ensemble import RandomForestClassifier
import sys

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import PROCESSED_DIR, RAW_DIR

# ==========================================
# 2. Trading strategy
# ==========================================
class MLStrategy(Strategy):
    def init(self):
        # The model generates signals ahead of time. The strategy only executes them.
        pass

    def next(self):
        # Read the most recent signal for the current session.
        if len(self.data.Pred_Signal) > 0:
            signal = self.data.Pred_Signal[-1]

            if signal == 1:
                # Buy when the model is bullish and no position is open.
                if not self.position:
                    self.buy()
            elif signal == 0:
                # Close the position when the model turns bearish.
                if self.position:
                    self.position.close()

# ==========================================
# 3. Core backtest function
# ==========================================
def backtester1(ticker="META"):
    print(f"Starting machine-learning backtest for {ticker}...")

    # --- A. Load the prepared dataset and raw price history ---
    csv_path = PROCESSED_DIR / f"{ticker}_sentiment_stock_merged.csv"
    price_path = RAW_DIR / f"{ticker}_history.csv"
    
    # Validate input files
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        print(f"Search path: {PROCESSED_DIR}")
        return
    if not os.path.exists(price_path):
        print(f"Error: file not found: {price_path}")
        print(f"Search path: {RAW_DIR}")
        return

    feature_df = pd.read_csv(csv_path)
    feature_df['date'] = pd.to_datetime(feature_df['date']).dt.tz_localize(None)
    feature_df = feature_df.sort_values('date').copy()

    price_df = pd.read_csv(price_path)
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
    price_df = price_df.sort_values('Date').copy()

    df = pd.merge(
        price_df,
        feature_df,
        left_on='Date',
        right_on='date',
        how='inner'
    )
    df = df.drop(columns=['date'])
    if 'Close_x' in df.columns:
        df.rename(columns={'Close_x': 'Close'}, inplace=True)
    if 'Close_y' in df.columns:
        df.drop(columns=['Close_y'], inplace=True)
    df.set_index('Date', inplace=True)
    
    print(f"Dataset ready: {len(df)} trading sessions")

    # --- B. Feature engineering ---
    df['Sentiment'] = df['adaptive_score']
    df['Returns'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Vol_5'] = df['Close'].rolling(window=5).std()
    df = df.dropna(subset=['Target']).copy()
    df['Target'] = df['Target'].astype(int)
    df.dropna(inplace=True)

    # --- C. Train model ---
    print("\nTraining model...")
    feature_cols = ['Sentiment', 'SMA_5', 'SMA_20', 'Returns', 'Vol_5']
    X = df[feature_cols]
    y = df['Target']

    # Use the last 30% of the series as the test segment.
    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Balanced class weights reduce bias toward the dominant class.
    model = RandomForestClassifier(n_estimators=100, min_samples_split=20, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Show feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances)

    # --- D. Generate signals ---
    # Generate signals only for the test window to avoid leakage.
    test_probs = model.predict_proba(X_test)[:, 1]
    backtest_data = df.iloc[split:].copy()
    backtest_data['Pred_Signal'] = (test_probs > 0.40).astype(int)
    
    print("\nSignal distribution (1=buy, 0=flat):")
    print(backtest_data['Pred_Signal'].value_counts())

    # --- E. Run backtest ---
    print(f"\nBacktest window: {backtest_data.index.min().date()} -> {backtest_data.index.max().date()}")
    
    # Initial cash is 10,000 USD with a 0.2% commission.
    bt = Backtest(backtest_data, MLStrategy, cash=10000, commission=.002)
    stats = bt.run()
    
    print("\nBacktest summary")
    print("-" * 32)
    print(stats)
    
    # Save the chart
    save_path = PROCESSED_DIR / f"{ticker}_ML_Final_Fixed.html"
    bt.plot(filename=str(save_path), open_browser=False)
    print(f"\nReport saved to: {save_path}")

# ==========================================
# 4. Standalone execution
# ==========================================
if __name__ == "__main__":
    backtester1("META")
