import feedparser
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta, date
import os
import time
import random
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


def _build_gdelt_query(ticker):
    """
    Build a broad but ticker-focused GDELT query.
    """
    alias_map = {
        "NVDA": ["NVIDIA", "Nvidia"],
        "META": ["Meta", "Facebook"],
        "AAPL": ["Apple"],
        "MSFT": ["Microsoft"],
        "GOOGL": ["Google", "Alphabet"],
        "AMZN": ["Amazon"],
        "TSLA": ["Tesla"],
        "AMD": ["Advanced Micro Devices", "AMD"],
    }
    terms = [ticker] + alias_map.get(ticker.upper(), [])
    quoted_terms = " OR ".join([f"\"{t}\"" for t in terms if isinstance(t, str) and t.strip()])
    market_context = "(stock OR shares OR earnings OR guidance OR revenue OR outlook OR AI OR semiconductor)"
    return f"({quoted_terms}) AND {market_context}"


def _google_query_terms(ticker):
    alias_map = {
        "NVDA": ["NVIDIA"],
        "META": ["Meta", "Facebook"],
        "AAPL": ["Apple"],
        "MSFT": ["Microsoft"],
        "GOOGL": ["Google", "Alphabet"],
        "AMZN": ["Amazon"],
        "TSLA": ["Tesla"],
        "AMD": ["Advanced Micro Devices"],
    }
    terms = [ticker] + alias_map.get(ticker.upper(), [])
    quoted = " OR ".join([f"\"{t}\"" for t in terms if isinstance(t, str) and t.strip()])
    return f"({quoted}) stock"


def _fetch_google_archive_news(ticker, start_dt, end_dt, window_days=30, sleep_sec=0.25):
    """
    Fetch historical headlines from Google News RSS using date-range query syntax.
    """
    query_base = _google_query_terms(ticker)
    all_rows = []
    cursor = start_dt
    total_windows = 0
    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=window_days - 1), end_dt)
        total_windows += 1

        # Google query syntax uses before:YYYY-MM-DD as exclusive upper bound.
        before_exclusive = window_end + timedelta(days=1)
        q = (
            f"{query_base} "
            f"after:{cursor.strftime('%Y-%m-%d')} "
            f"before:{before_exclusive.strftime('%Y-%m-%d')}"
        )
        params = {
            "q": q,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        try:
            resp = requests.get("https://news.google.com/rss/search", params=params, timeout=20)
            if resp.status_code != 200:
                print(f"   Google archive {cursor}..{window_end} failed: HTTP {resp.status_code}")
            else:
                feed = feedparser.parse(resp.content)
                entries = getattr(feed, "entries", [])
                if total_windows % 6 == 0 or len(entries) > 0:
                    print(f"   Google archive {cursor}..{window_end}: {len(entries)} entries")
                for entry in entries:
                    title_text = getattr(entry, 'title', '')
                    link = getattr(entry, 'link', '')
                    if not isinstance(title_text, str) or not title_text.strip():
                        continue
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        date_str = time.strftime('%Y-%m-%d', entry.published_parsed)
                    else:
                        date_str = cursor.strftime('%Y-%m-%d')
                    all_rows.append({
                        'date': date_str,
                        'title': title_text.strip(),
                        'source': 'Google_Archive',
                        'link': link
                    })
        except Exception as e:
            print(f"   Google archive {cursor}..{window_end} error: {e}")

        cursor = window_end + timedelta(days=1)
        time.sleep(sleep_sec)

    print(f"   Google archive scanned {total_windows} windows, collected {len(all_rows)} raw rows")
    return all_rows


def _parse_gdelt_article_date(article):
    """
    Parse date from GDELT article payload.
    """
    raw = article.get('seendate') or article.get('date')
    if isinstance(raw, str):
        raw = raw.strip()
        if len(raw) >= 8 and raw[:8].isdigit():
            try:
                return datetime.strptime(raw[:8], "%Y%m%d").strftime("%Y-%m-%d")
            except Exception:
                pass
    return None


def _fetch_gdelt_news(
    ticker,
    start_dt,
    end_dt,
    max_records=250,
    window_days=365,
    sleep_sec=1.2,
    max_retries=4
):
    """
    Fetch historical headlines from GDELT in rolling time windows.
    """
    query = _build_gdelt_query(ticker)
    all_rows = []
    cursor = start_dt
    total_windows = 0
    while cursor <= end_dt:
        window_end = min(cursor + timedelta(days=window_days - 1), end_dt)
        total_windows += 1

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": int(max_records),
            "sort": "datedesc",
            "startdatetime": cursor.strftime("%Y%m%d000000"),
            "enddatetime": window_end.strftime("%Y%m%d235959"),
        }
        success = False
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params=params,
                    timeout=20
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    articles = payload.get("articles", []) if isinstance(payload, dict) else []
                    print(f"   GDELT {cursor}..{window_end}: {len(articles)} articles")
                    for art in articles:
                        title = art.get("title")
                        if not isinstance(title, str) or not title.strip():
                            continue
                        link = art.get("url", "")
                        date_str = _parse_gdelt_article_date(art) or cursor.strftime("%Y-%m-%d")
                        all_rows.append({
                            "date": date_str,
                            "title": title.strip(),
                            "source": "GDELT",
                            "link": link
                        })
                    success = True
                    break

                if resp.status_code == 429:
                    backoff = min(20.0, 2.0 * (2 ** attempt))
                    print(
                        f"   GDELT {cursor}..{window_end} rate-limited (attempt {attempt + 1}/{max_retries}), "
                        f"sleeping {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue

                print(
                    f"   GDELT {cursor}..{window_end} failed: HTTP {resp.status_code} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(1.5)
            except Exception as e:
                backoff = min(20.0, 2.0 * (2 ** attempt))
                print(
                    f"   GDELT {cursor}..{window_end} parse/request error "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(backoff)

        if not success:
            print(f"   GDELT {cursor}..{window_end}: no usable payload after retries")

        cursor = window_end + timedelta(days=1)
        time.sleep(sleep_sec)

    print(f"   GDELT scanned {total_windows} windows, collected {len(all_rows)} raw rows")
    return all_rows

# ==========================================
# 2. Core function used by main.py
# ==========================================
def run_news_collector(
    ticker="META",
    history_start="2014-01-01",
    history_end=None,
    use_google_archive=True,
    use_gdelt=False
):
    print(f"Starting news collection for {ticker}...", flush=True)
    
    all_news = []
    
    # --- A. Primary source: yfinance API ---
    print("[1/4] Requesting Yahoo Finance API...", flush=True)
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
        print(f"   Yahoo API returned {len(yf_news)} items", flush=True)
    except Exception as e:
        print(f"   Yahoo API request failed: {e}", flush=True)

    # --- B. Historical source: Google archive ---
    print("[2/5] Pulling historical archive from Google RSS...", flush=True)
    try:
        start_dt = datetime.strptime(str(history_start), "%Y-%m-%d").date()
    except Exception:
        start_dt = date(2014, 1, 1)
    if history_end:
        try:
            end_dt = datetime.strptime(str(history_end), "%Y-%m-%d").date()
        except Exception:
            end_dt = datetime.utcnow().date()
    else:
        end_dt = datetime.utcnow().date()

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    if use_google_archive:
        try:
            google_rows = _fetch_google_archive_news(
                ticker,
                start_dt=start_dt,
                end_dt=end_dt,
                window_days=30,
                sleep_sec=0.25
            )
            all_news.extend(google_rows)
        except Exception as e:
            print(f"   Google archive fetch failed: {e}", flush=True)

    # --- C. Optional historical source: GDELT archive ---
    print("[3/5] Pulling historical archive from GDELT (optional)...", flush=True)
    if use_gdelt:
        try:
            gdelt_rows = _fetch_gdelt_news(
                ticker,
                start_dt=start_dt,
                end_dt=end_dt,
                max_records=250,
                window_days=365,
                sleep_sec=1.2,
                max_retries=4
            )
            all_news.extend(gdelt_rows)
        except Exception as e:
            print(f"   GDELT fetch failed: {e}", flush=True)
    else:
        print("   Skipped (use_gdelt=False)", flush=True)

    # --- D. Secondary source: live RSS feeds ---
    rss_sources = [
        {"name": "Google News", "url": f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"},
        {"name": "CNBC Finance", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"},
        {"name": "Investing.com", "url": "https://www.investing.com/rss/news_25.rss"}
    ]
    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    ]
    print("[4/5] Scanning RSS feeds...", flush=True)
    
    for source in rss_sources:
        try:
            headers = {"User-Agent": random.choice(user_agents)}
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
            print(f"   RSS source error for {source['name']}: {e}", flush=True)

    # --- E. Merge and save ---
    print("[5/5] Merging historical and newly collected data...", flush=True)
    
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
        print(f"Archive file found: {history_file_path}", flush=True)
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
                print(f"   Loaded {len(master_df)} archived rows", flush=True)
            else:
                print("   Archive format error: required columns are missing", flush=True)
        except Exception as e:
            print(f"   Failed to read archive file: {e}", flush=True)
    else:
        print(f"   Archive file not found at {history_file_path}. Skipping.", flush=True)

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
        print(f"Saved combined dataset to: {standard_save_path}", flush=True)
        print(f"Total news rows: {len(master_df)}", flush=True)
    else:
        print("No data available to save.", flush=True)

    print(f"News collection finished for {ticker}.", flush=True)

# ==========================================
# 3. Standalone execution
# ==========================================
if __name__ == "__main__":
    run_news_collector("META")
