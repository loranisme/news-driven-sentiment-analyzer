import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import sys

# ==========================================
# 1. 🟢 路径配置 (防止找不到 src.config)
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR 

def collect_stock_history(ticker="AAPL", period="5y"):
    """
    下载并清洗股价历史数据
    :param ticker: 股票代码
    :param period: 下载时长 (1y, 2y, 5y, max)
    """
    print(f"📉 [Price Collector] 正在下载 {ticker} 过去 {period} 的数据...")

    try:
        # 1. 下载数据
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

        if df.empty:
            print(f"❌ 下载失败：未获取到 {ticker} 的数据。")
            return

        # 2. 清洗数据 (处理 MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 重置索引
        df = df.reset_index()

        # 3. 格式化日期
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = df['Date'].dt.tz_localize(None) 
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # 4. 筛选列
        needed_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in needed_cols if c in df.columns]
        df = df[available_cols]

        # 5. 保存
        # 确保目录存在
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # 动态生成文件名
        save_path = RAW_DIR / f"{ticker}_history.csv"
        
        df.to_csv(save_path, index=False)
        
        print(f"✅ 股价数据已保存！")
        print(f"📍 路径: {save_path}")
        print(f"📊 范围: {df['Date'].min()} 到 {df['Date'].max()} (共 {len(df)} 条)")

    except Exception as e:
        print(f"❌ 股价下载发生严重错误: {e}")

# ==========================================
# 2. 🔌 定义标准接口 (供 main.py 调用)
# ==========================================
def run_price_collector(ticker="META"):
    # 为了回测尽量准确，我们默认下载 'max' (所有历史数据)
    collect_stock_history(ticker=ticker, period="max")

if __name__ == "__main__":
    run_price_collector("META")