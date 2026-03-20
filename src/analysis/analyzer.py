import pandas as pd
import numpy as np
from textblob import TextBlob
import os
import sys
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR, PROCESSED_DIR

# ==========================================
# 2. Core analyzer
# ==========================================
class AdaptiveSentimentAnalyzer:
    def __init__(self, ticker="AAPL"):
        self.ticker = ticker
        
        # Dynamic input paths
        self.news_path = RAW_DIR / f"{ticker}_news.csv"
        self.stock_path = RAW_DIR / f"{ticker}_history.csv"
        
        # Dynamic output paths
        self.output_sentiment_path = PROCESSED_DIR / f"{ticker}_sentiment_daily.csv"
        self.final_merged_path = PROCESSED_DIR / f"{ticker}_sentiment_stock_merged.csv"
        
        # Ensure the output directory exists.
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def _calculate_base_sentiment(self, text):
        if not isinstance(text, str):
            return 0
        return TextBlob(text).sentiment.polarity

    def _add_technical_indicators(self, df):
        """
        Compute technical indicators such as moving averages, RSI, and momentum.
        """
        # Ensure the calculations use the Close column.
        if 'Close' not in df.columns:
            # Promote lowercase close to Close when needed.
            if 'close' in df.columns:
                df['Close'] = df['close']
            else:
                print("Warning: Close column not found. Technical indicators may be incomplete.")
                return df

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Trend_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
        df['Momentum_5D'] = df['Close'].pct_change(periods=5)

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill initial gaps from rolling calculations.
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def process_news(self):
        print(f"[Analyzer] Reading {self.ticker} news from {self.news_path}...")
        
        if not os.path.exists(self.news_path):
            print(f"Error: file not found: {self.news_path}")
            print("Run data collection first.")
            return None

        # Read the CSV file
        df = pd.read_csv(self.news_path)
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'date' not in df.columns or 'title' not in df.columns:
             print(f"Error: CSV is missing date or title columns. Current columns: {df.columns.tolist()}")
             return None

        # Normalize dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.date
        df.dropna(subset=['date'], inplace=True)

        print("Calculating sentiment scores...")
        tqdm.pandas(desc="Sentiment Progress")
        df['sentiment'] = df['title'].progress_apply(self._calculate_base_sentiment)

        daily_df = df.groupby('date')['sentiment'].mean().reset_index()
        daily_df.columns = ['date', 'sentiment_score']
        
        return daily_df

    def apply_adaptive_learning(self, daily_df, window=20):
        print("Calculating adaptive sentiment features...")
        daily_df['rolling_mean'] = daily_df['sentiment_score'].rolling(window=window).mean()
        daily_df['rolling_std'] = daily_df['sentiment_score'].rolling(window=window).std()
        
        daily_df['sentiment_z_score'] = (
            (daily_df['sentiment_score'] - daily_df['rolling_mean']) / daily_df['rolling_std']
        ).fillna(0)

        daily_df['adaptive_score'] = daily_df['sentiment_z_score']
        return daily_df

    def merge_with_price(self, sentiment_df):
        print(f"Merging sentiment with {self.ticker} price history...")
        
        if not os.path.exists(self.stock_path):
            print(f"Error: price file not found: {self.stock_path}")
            print("Run price collection first.")
            return None

        price_df = pd.read_csv(self.stock_path)
        
        # Normalize column names
        price_df.columns = [c.lower().strip() for c in price_df.columns]
        
        # Validate required date column
        if 'date' not in price_df.columns:
            print("Error: date column not found in price file.")
            return None
            
        # Validate required close column
        if 'close' in price_df.columns:
            price_df.rename(columns={'close': 'Close'}, inplace=True)
        elif 'Close' not in price_df.columns:
            print("Error: close column not found in price file.")
            return None

        # Normalize dates
        price_df['date'] = pd.to_datetime(price_df['date'], utc=True).dt.date
        
        # Keep all trading days and carry the latest sentiment forward.
        merged_df = pd.merge(price_df, sentiment_df, on='date', how='left')
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        merged_df['sentiment_score'] = merged_df['sentiment_score'].ffill().fillna(0)
        merged_df['adaptive_score'] = merged_df['adaptive_score'].ffill().fillna(0)

        # Add technical features
        merged_df = self._add_technical_indicators(merged_df)

        # Create the next-session target label
        merged_df['next_close'] = merged_df['Close'].shift(-1)
        merged_df['Target'] = (merged_df['next_close'] > merged_df['Close']).astype(float)
        merged_df = merged_df.dropna(subset=['next_close']).copy()
        merged_df['Target'] = merged_df['Target'].astype(int)
        
        # Keep the final training columns
        cols_to_keep = [
            'date', 'sentiment_score', 'adaptive_score', 
            'Trend_Signal', 'RSI', 'Momentum_5D', 
            'Close', 'Target'
        ]
        final_cols = [c for c in cols_to_keep if c in merged_df.columns]
        
        return merged_df[final_cols]

    def run(self):
        daily_df = self.process_news()
        if daily_df is None: return

        learned_df = self.apply_adaptive_learning(daily_df)
        learned_df.to_csv(self.output_sentiment_path, index=False)
        final_df = self.merge_with_price(learned_df)
        
        if final_df is not None and not final_df.empty:
            final_df.to_csv(self.final_merged_path, index=False)
            print(f"Training dataset created: {self.final_merged_path}")
            print("Included fields: sentiment, technical features, and target labels")
        else:
            print("Warning: merged dataset is empty. Check whether the news and price dates overlap.")

# ==========================================
# 3. Public entry point for main.py
# ==========================================
def run_sentiment_analysis(ticker="META"):
    # Instantiate and run the analyzer.
    analyzer = AdaptiveSentimentAnalyzer(ticker=ticker)
    analyzer.run()

if __name__ == "__main__":
    run_sentiment_analysis("META")
