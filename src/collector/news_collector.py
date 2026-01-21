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
# 1. 🟢 路径配置 (确保能找到 src.config)
# ==========================================
# 这一步是为了防止单独运行此文件时找不到模块
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR

# ==========================================
# 2. 🔌 定义核心函数 (供 main.py 调用)
# ==========================================
def run_news_collector(ticker="META"):
    print(f"🚀 启动全网新闻采集器，目标: {ticker} ...")
    
    all_news = []
    
    # --- A. 第一梯队：yfinance 官方 API ---
    print("📡 [1/3] 正在请求 Yahoo Finance API...")
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
        print(f"   ✅ Yahoo API 获取到 {len(yf_news)} 条")
    except Exception as e:
        print(f"   ❌ Yahoo API 失败: {e}")

    # --- B. 第二梯队：RSS 轮询 ---
    rss_sources = [
        {"name": "Google News", "url": f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"},
        {"name": "CNBC Finance", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"},
        {"name": "Investing.com", "url": "https://www.investing.com/rss/news_25.rss"}
    ]
    ua = UserAgent()
    print("📡 [2/3] 正在扫描 RSS 数据源...")
    
    for source in rss_sources:
        try:
            headers = {"User-Agent": ua.random}
            response = requests.get(source['url'], headers=headers, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                for entry in feed.entries:
                    summary_text = getattr(entry, 'summary', '')
                    title_text = getattr(entry, 'title', '')
                    
                    # 过滤非相关新闻 (Google News 除外，因为它搜索的就是关键词)
                    if source['name'] != "Google News":
                        if ticker not in title_text and ticker not in summary_text:
                            continue
                    
                    # 处理时间
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
            print(f"   ⚠️ RSS 源 {source['name']} 报错: {e}")

    # --- C. 数据整合与保存 ---
    print("🧹 [3/3] 正在整合历史数据与新数据...")
    
    new_df = pd.DataFrame(all_news)
    if not new_df.empty:
        new_df['date'] = pd.to_datetime(new_df['date'])
        new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
    else:
        new_df = pd.DataFrame(columns=['date', 'title', 'source'])

    # 定义路径
    history_file_path = RAW_DIR / "allnews.csv"
    standard_save_path = RAW_DIR / f"{ticker}_news.csv"
    
    master_df = pd.DataFrame()

    # 1. 优先读取历史归档 (allnews.csv)
    if os.path.exists(history_file_path):
        print(f"📖 发现历史归档文件: {history_file_path}")
        try:
            hist_df = pd.read_csv(history_file_path)
            
            # 清洗列名
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
                print(f"   ✅ 成功加载历史数据: {len(master_df)} 条")
            else:
                print(f"   ❌ 格式错误: allnews.csv 缺少必要列")
        except Exception as e:
            print(f"   ❌ 读取历史文件出错: {e}")
    else:
        print(f"   ℹ️ 未找到 {history_file_path} (如果不在此处放置历史大文件，可忽略)。")

    # 2. 读取之前生成的标准文件 (防止重复覆盖)
    if os.path.exists(standard_save_path):
        try:
            prev_df = pd.read_csv(standard_save_path)
            master_df = pd.concat([master_df, prev_df], ignore_index=True)
        except:
            pass

    # 3. 合并新爬取的数据
    if not new_df.empty:
        master_df = pd.concat([master_df, new_df], ignore_index=True)

    # 4. 去重并保存
    if not master_df.empty:
        master_df.sort_values(by='date', inplace=True)
        # 关键去重：同一天、同一个标题只保留一条
        master_df.drop_duplicates(subset=['title', 'date'], keep='last', inplace=True)
        
        master_df.to_csv(standard_save_path, index=False)
        print(f"💾 [最终完成] 数据已保存至: {standard_save_path}")
        print(f"📊 总计包含新闻: {len(master_df)} 条")
    else:
        print("⚠️ 没有任何数据可保存。")

    print(f"✅ {ticker} 新闻采集任务结束！")

# ==========================================
# 3. 🏁 独立运行开关
# ==========================================
if __name__ == "__main__":
    run_news_collector("META")