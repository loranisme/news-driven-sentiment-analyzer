import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from backtesting import Backtest, Strategy
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 🟢 配置路径 (修复了之前的报错)
# ==========================================
# 获取当前脚本的根目录
BASE_DIR = Path(os.getcwd())
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ==========================================
# 2. 🧠 定义 AI 交易策略类
# ==========================================
class MLStrategy(Strategy):
    def init(self):
        # 这里的逻辑很简单：AI 已经把买卖信号算好了，我们只负责执行
        pass

    def next(self):
        # 获取当前这一天的 AI 信号 (1=看涨/买入, 0=看跌/卖出)
        # 注意：Backtesting.py 的 self.data 是个数组，[-1] 代表"今天"
        if len(self.data.Pred_Signal) > 0:
            signal = self.data.Pred_Signal[-1]

            if signal == 1:
                # 如果 AI 说涨，且我们没仓位，就全仓买入
                if not self.position:
                    self.buy()
            elif signal == 0:
                # 如果 AI 说跌，且我们要有仓位，就赶紧卖掉
                if self.position:
                    self.position.close()

# ==========================================
# 3. 🚀 核心函数 (已改名为 backtester1)
# ==========================================
def backtester1(ticker="META"):
    print(f"🧠 启动 {ticker} 机器学习回测管道 (函数名: backtester1)...")

    # --- A. 加载情绪数据 ---
    csv_path = PROCESSED_DIR / f"{ticker}_sentiment_stock_merged.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        print(f"当前搜索路径: {PROCESSED_DIR}")
        return

    df_sent = pd.read_csv(csv_path)
    # 清理列名和格式
    df_sent.columns = [c.lower().strip() for c in df_sent.columns]
    df_sent['date'] = pd.to_datetime(df_sent['date']).dt.tz_localize(None)
    df_sent['date_str'] = df_sent['date'].dt.strftime('%Y-%m-%d')
    
    # 按天聚合情绪分
    df_sent_grouped = df_sent.groupby('date_str')['adaptive_score'].mean().reset_index()

    # --- B. 加载股价数据 (获取完整时间线) ---
    print("☁️ 从 Yahoo 下载完整行情数据...")
    df_price = yf.download(ticker, start="2019-01-01", progress=False)
    
    # 处理 MultiIndex 列名问题 (Yahoo 的新特性)
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
    
    df_price = df_price.reset_index()
    df_price['date_str'] = pd.to_datetime(df_price['Date']).dt.strftime('%Y-%m-%d')

    # --- C. 数据对齐 (关键修复：Left Join + Forward Fill) ---
    print("🔗 正在合并数据 (保留所有交易日)...")
    
    # 1. 使用 Left Join，保留每一天的股价，不管有没有新闻
    df = pd.merge(df_price, df_sent_grouped, on='date_str', how='left')
    
    # 2. 核心逻辑：情绪惯性 (Forward Fill)
    # 如果今天没新闻，就假设情绪和昨天一样
    df['adaptive_score'] = df['adaptive_score'].ffill()
    df['adaptive_score'] = df['adaptive_score'].fillna(0) # 开头填0
    
    df['Date'] = pd.to_datetime(df['date_str'])
    df.set_index('Date', inplace=True)
    
    print(f"✅ 数据准备完毕: 共 {len(df)} 个连续交易日")

    # --- D. 特征工程 ---
    df['Sentiment'] = df['adaptive_score']
    df['Returns'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Vol_5'] = df['Close'].rolling(window=5).std()
    
    # 目标：明天涨跌 (1=涨, 0=跌)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)

    # --- E. 训练模型 ---
    print("\n🤖 正在训练 AI...")
    feature_cols = ['Sentiment', 'SMA_5', 'SMA_20', 'Returns', 'Vol_5']
    X = df[feature_cols]
    y = df['Target']

    # 训练/测试切分 (后 30% 做测试)
    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 增加 balanced 权重，防止 AI 偷懒全猜跌
    model = RandomForestClassifier(n_estimators=100, min_samples_split=20, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # 打印因子重要性
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n🔍 因子重要性排名:")
    print(importances)

    # --- F. 生成信号 ---
    # 使用概率预测：如果上涨概率 > 40% 就买
    probs = model.predict_proba(X)[:, 1]
    df['Pred_Signal'] = (probs > 0.40).astype(int)
    
    print("\n🔍 AI 交易信号分布 (1=买, 0=卖):")
    print(df['Pred_Signal'].value_counts())

    # --- G. 执行回测 ---
    backtest_data = df.iloc[split:].copy()
    print(f"\n🏃‍♂️ 开始回测区间: {backtest_data.index.min().date()} -> {backtest_data.index.max().date()}")
    
    # 初始资金 10000 美元，佣金千分之二
    bt = Backtest(backtest_data, MLStrategy, cash=10000, commission=.002)
    stats = bt.run()
    
    print("\n💰 === 最终成绩单 ===")
    print(stats)
    
    # 保存图表
    save_path = PROCESSED_DIR / f"{ticker}_ML_Final_Fixed.html"
    bt.plot(filename=str(save_path), open_browser=False)
    print(f"\n📊 图表已保存: {save_path}")

# ==========================================
# 4. 🏁 启动开关
# ==========================================
if __name__ == "__main__":
    # 调用改名后的函数
    backtester1("META")