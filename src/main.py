import sys
import os
import time
from pathlib import Path

# ==========================================
# 1. Path setup
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DEFAULT_TICKER

print(f"Project root: {project_root}")

# ==========================================
# 2. Dynamic module loading
# ==========================================
modules = {}

# --- Data collection ---
try:
    from src.collector.news_collector import run_news_collector
    modules['news_collector'] = run_news_collector
except ImportError: pass

try:
    from src.collector.price_collector import run_price_collector
    modules['price_collector'] = run_price_collector
except ImportError: pass

# --- Analysis ---
try:
    from src.analysis.analyzer import run_sentiment_analysis
    modules['analyzer'] = run_sentiment_analysis
except ImportError: pass

# --- Training ---
try:
    from models.sentiment_trend_pro import run_train_model
    modules['trainer'] = run_train_model
    print("Loaded module: Trainer")
except ImportError as e:
    print(f"Module not loaded: Trainer ({e})")

# --- Backtesting ---
try:
    from src.strategy.backtester1 import backtester1
    modules['backtester'] = backtester1
except ImportError: pass

# --- Forecasting ---
try:
    from src.prediction.forecast_tech import run_tech_forecast
    modules['forecaster'] = run_tech_forecast
    print("Loaded module: Forecaster")
except ImportError as e:
    print(f"Module not loaded: Forecaster ({e})")


# ==========================================
# 3. Main console
# ==========================================
def main():
    default_ticker = DEFAULT_TICKER
    
    while True:
        print("\n" + "=" * 64)
        print("   AI Quant Research Console")
        print(f"   Active ticker: {default_ticker}")
        print("=" * 64)
        
        print("1. Collect data")
        print("2. Run sentiment analysis")
        print("3. Train model")
        print("4. Run backtest")
        print("-" * 30)
        print("5. Generate next-session forecast")
        print("-" * 30)
        print("9. Run full pipeline (1 -> 2 -> 3 -> 4 -> 5)")
        print("-" * 30)
        print("C. Change ticker")
        print("0. Exit")
        
        choice = input("\nEnter command: ").upper().strip()

        # --- 1. Collect data ---
        if choice == "1":
            print("\nStarting data collection...")
            if 'news_collector' in modules: modules['news_collector'](default_ticker)
            if 'price_collector' in modules: modules['price_collector'](default_ticker)

        # --- 2. Analysis ---
        elif choice == "2":
            print("\nRunning sentiment analysis...")
            if 'analyzer' in modules: modules['analyzer'](default_ticker)

        # --- 3. Training ---
        elif choice == "3":
            print("\nTraining model...")
            if 'trainer' in modules:
                modules['trainer'](default_ticker)
            else:
                print("Error: training module not found")

        # --- 4. Backtesting ---
        elif choice == "4":
            print("\nRunning backtest...")
            if 'backtester' in modules: modules['backtester'](default_ticker)

        # --- 5. Forecasting ---
        elif choice == "5":
            print(f"\nGenerating forecast for {default_ticker}...")
            if 'forecaster' in modules:
                modules['forecaster'](default_ticker)
            else:
                print("Error: forecast module not found")

        # --- 9. Full pipeline ---
        elif choice == "9":
            print(f"\nRunning full pipeline for {default_ticker}...")
            
            # Step 1
            if 'news_collector' in modules: modules['news_collector'](default_ticker)
            if 'price_collector' in modules: modules['price_collector'](default_ticker)
            time.sleep(1)
            
            # Step 2
            if 'analyzer' in modules: 
                modules['analyzer'](default_ticker)
                time.sleep(1)
            
            # Step 3
            if 'trainer' in modules:
                print("\n[Step 3] Training model...")
                modules['trainer'](default_ticker)
                time.sleep(1)

            # Step 4
            if 'backtester' in modules:
                print("\n[Step 4] Running backtest...")
                modules['backtester'](default_ticker)
            
            # Step 5
            if 'forecaster' in modules:
                 print("\n[Step 5] Generating forecast...")
                 modules['forecaster'](default_ticker)
            
            print("\nPipeline completed.")

        # --- Other actions ---
        elif choice == "C":
            new_ticker = input("Enter a new ticker (for example NVDA): ").upper()
            if new_ticker: default_ticker = new_ticker
        elif choice == "0":
            print("\nSession closed.")
            break
        else:
            print("\nInvalid command.")

if __name__ == "__main__":
    main()
