import pandas as pd
import numpy as np
from textblob import TextBlob
import os
import sys
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. 🟢 路径配置 (确保能找到 src.config)
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR, PROCESSED_DIR

# ==========================================
# 2. 🧠 定义核心类
# ==========================================
class AdaptiveSentimentAnalyzer:
    def __init__(self, ticker="AAPL"):
        self.ticker = ticker
        
        # 1. 动态输入路径
        self.news_path = RAW_DIR / f"{ticker}_news.csv"
        self.stock_path = RAW_DIR / f"{ticker}_history.csv"
        
        # 2. 动态输出路径
        self.output_sentiment_path = PROCESSED_DIR / f"{ticker}_sentiment_daily.csv"
        self.final_merged_path = PROCESSED_DIR / f"{ticker}_sentiment_stock_merged.csv"
        
        # 确保输出目录存在
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def _calculate_base_sentiment(self, text):
        if not isinstance(text, str):
            return 0
        return TextBlob(text).sentiment.polarity

    def _add_technical_indicators(self, df):
        """
        📊 计算技术指标 (MA, RSI, Momentum)
        """
        # 确保操作的是 'Close' 列
        if 'Close' not in df.columns:
            # 如果只有小写 close，复制一份给 Close
            if 'close' in df.columns:
                df['Close'] = df['close']
            else:
                print("⚠️ 警告：找不到 Close 列，技术指标计算可能会失败。")
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
        
        # 填充 NaN
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def process_news(self):
        print(f"📖 [Analyzer] 正在读取 {self.ticker} 新闻: {self.news_path}...")
        
        if not os.path.exists(self.news_path):
            print(f"❌ 找不到文件: {self.news_path}")
            print(f"💡 请先运行 1.数据采集 (News Collector)！")
            return None

        # 读取 CSV
        df = pd.read_csv(self.news_path)
        
        # 清洗列名
        df.columns = [c.lower().strip() for c in df.columns]
        
        if 'date' not in df.columns or 'title' not in df.columns:
             print(f"❌ CSV 缺少 date 或 title 列。当前列名: {df.columns.tolist()}")
             return None

        # 处理日期
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.date
        df.dropna(subset=['date'], inplace=True)

        print("⚡ 正在计算情绪分数...")
        tqdm.pandas(desc="Sentiment Progress")
        df['sentiment'] = df['title'].progress_apply(self._calculate_base_sentiment)

        daily_df = df.groupby('date')['sentiment'].mean().reset_index()
        daily_df.columns = ['date', 'sentiment_score']
        
        return daily_df

    def apply_adaptive_learning(self, daily_df, window=20):
        print(f"🧠 正在计算自适应情绪指标...")
        daily_df['rolling_mean'] = daily_df['sentiment_score'].rolling(window=window).mean()
        daily_df['rolling_std'] = daily_df['sentiment_score'].rolling(window=window).std()
        
        daily_df['sentiment_z_score'] = (
            (daily_df['sentiment_score'] - daily_df['rolling_mean']) / daily_df['rolling_std']
        ).fillna(0)

        daily_df['adaptive_score'] = daily_df['sentiment_z_score']
        return daily_df

    def merge_with_price(self, sentiment_df):
        print(f"📉 正在与 {self.ticker} 股价历史合并...")
        
        if not os.path.exists(self.stock_path):
            print(f"❌ 找不到股价文件: {self.stock_path}")
            print("💡 请先运行 1.数据采集 (Price Collector)！")
            return None

        price_df = pd.read_csv(self.stock_path)
        
        # 清洗列名
        price_df.columns = [c.lower().strip() for c in price_df.columns]
        
        # 确保有 date 列
        if 'date' not in price_df.columns:
            print(f"❌ 错误：股价文件中找不到 'date' 列。")
            return None
            
        # 确保有 close 列
        if 'close' in price_df.columns:
            price_df.rename(columns={'close': 'Close'}, inplace=True)
        elif 'Close' not in price_df.columns:
            print(f"❌ 错误：股价文件中找不到 'close' 列。")
            return None

        # 转换日期格式
        price_df['date'] = pd.to_datetime(price_df['date'], utc=True).dt.date
        
        # 计算技术指标
        price_df = self._add_technical_indicators(price_df)
        
        # 制作标签 Target (预测明天涨跌)
        price_df['price_change'] = price_df['Close'].diff().shift(-1)
        price_df['Target'] = (price_df['price_change'] > 0).astype(int)
        
        # 合并
        merged_df = pd.merge(sentiment_df, price_df, on='date', how='inner')
        
        # 筛选最终需要的列
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

        # 保存中间结果
        daily_df.to_csv(self.output_sentiment_path, index=False)
        
        learned_df = self.apply_adaptive_learning(daily_df)
        final_df = self.merge_with_price(learned_df)
        
        if final_df is not None and not final_df.empty:
            final_df.to_csv(self.final_merged_path, index=False)
            print(f"🚀 [成功] {self.ticker} 训练集已生成: {self.final_merged_path}")
            print(f"📊 包含情绪 + 技术指标 + 预测目标")
        else:
            print("⚠️ 合并结果为空！请检查新闻日期和股价日期是否有重叠。")

# ==========================================
# 3. 🔌 定义标准接口 (供 main.py 调用)
# ==========================================
def run_sentiment_analysis(ticker="META"):
    # 实例化类并运行
    analyzer = AdaptiveSentimentAnalyzer(ticker=ticker)
    analyzer.run()

if __name__ == "__main__":
    run_sentiment_analysis("META")