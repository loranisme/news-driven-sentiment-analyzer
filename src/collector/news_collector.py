import feedparser
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
import os
import time
from fake_useragent import UserAgent
from pathlib import Path

# ==========================================
# 1. Path setup
# ==========================================
# This keeps imports working when the file is executed directly.
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR

# ==========================================
# 2. Core function used by main.py
# ==========================================
def run_news_collector(ticker="META"):
    print(f"Starting news collection for {ticker}...")
    
    all_news = []
    
    # --- A. Primary source: yfinance API ---
    print("[1/3] Requesting Yahoo Finance API...")
    try:
        stock = yf.Ticker(ticker)
        yf_news = stock.news
        for item in yf_news:
            ts = item.get('providerPublishTime', time.time())
            date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            all_news.append({
                'date': date_str,
                'title': item.get('title'),
                'source': 'Yahoo_API',
                'link': item.get('link')
            })
        print(f"   Yahoo API returned {len(yf_news)} items")
    except Exception as e:
        print(f"   Yahoo API request failed: {e}")

    # --- B. Secondary source: RSS feeds ---
    rss_sources = [
        {"name": "Google News", "url": f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"},
        {"name": "CNBC Finance", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"},
        {"name": "Investing.com", "url": "https://www.investing.com/rss/news_25.rss"}
    ]
    ua = UserAgent()
    print("[2/3] Scanning RSS feeds...")
    
    for source in rss_sources:
        try:
            headers = {"User-Agent": ua.random}
            response = requests.get(source['url'], headers=headers, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries:
                    summary_text = getattr(entry, 'summary', '')
                    title_text = getattr(entry, 'title', '')
                    
                    # Filter unrelated news for broad feeds.
                    if source['name'] != "Google News":
                        if ticker not in title_text and ticker not in summary_text:
                            continue
                    
                    # Normalize the published date.
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        date_str = time.strftime('%Y-%m-%d', entry.published_parsed)
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')

                    all_news.append({
                        'date': date_str,
                        'title': title_text,
                        'source': source['name'],
                        'link': entry.link
                    })
        except Exception as e:
            print(f"   RSS source error for {source['name']}: {e}")

    # --- C. Merge and save ---
    print("[3/3] Merging historical and newly collected data...")
    
    new_df = pd.DataFrame(all_news)
    if not new_df.empty:
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
    else:
        new_df = pd.DataFrame(columns=['date', 'title', 'source'])

    # Define paths
    history_file_path = RAW_DIR / "allnews.csv"
    standard_save_path = RAW_DIR / f"{ticker}_news.csv"
    
    master_df = pd.DataFrame()

    # 1. Load optional archive file first.
    if os.path.exists(history_file_path):
        print(f"Archive file found: {history_file_path}")
        try:
            hist_df = pd.read_csv(history_file_path)
            
            # Normalize column names
            hist_df.columns = [c.lower().strip() for c in hist_df.columns]
            rename_map = {
                'headline': 'title',
                'news': 'title',
                'time': 'date',
                'timestamp': 'date'
            }
            hist_df.rename(columns=rename_map, inplace=True)
            
            if 'date' in hist_df.columns and 'title' in hist_df.columns:
                hist_df['date'] = pd.to_datetime(hist_df['date'], utc=True, errors='coerce')
                hist_df = hist_df.dropna(subset=['date'])
                hist_df['date'] = hist_df['date'].dt.strftime('%Y-%m-%d')
                
                if 'source' not in hist_df.columns:
                    hist_df['source'] = 'Historical_Archive'
                
                master_df = hist_df
                print(f"   Loaded {len(master_df)} archived rows")
            else:
                print("   Archive format error: required columns are missing")
        except Exception as e:
            print(f"   Failed to read archive file: {e}")
    else:
        print(f"   Archive file not found at {history_file_path}. Skipping.")

    # 2. Load the existing ticker file to avoid overwriting past runs.
    if os.path.exists(standard_save_path):
        try:
            prev_df = pd.read_csv(standard_save_path)
            master_df = pd.concat([master_df, prev_df], ignore_index=True)
        except:
            pass

    # 3. Append newly collected rows
    if not new_df.empty:
        master_df = pd.concat([master_df, new_df], ignore_index=True)

    # 4. Deduplicate and save
    if not master_df.empty:
        master_df.sort_values(by='date', inplace=True)
        # Keep only one record for the same title on the same day.
        master_df.drop_duplicates(subset=['title', 'date'], keep='last', inplace=True)
        
        master_df.to_csv(standard_save_path, index=False)
        print(f"Saved combined dataset to: {standard_save_path}")
        print(f"Total news rows: {len(master_df)}")
    else:
        print("No data available to save.")

    print(f"News collection finished for {ticker}.")

# ==========================================
# 3. Standalone execution
# ==========================================
if __name__ == "__main__":
    run_news_collector("META")
