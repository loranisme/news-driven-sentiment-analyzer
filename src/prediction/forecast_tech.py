import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ==========================================
# 1. 🟢 路径配置 (防止找不到 src.config)
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DIR

# ==========================================
# 2. 🔮 定义核心类
# ==========================================
class TechForecaster:
    def __init__(self, ticker="TSLA"):
        self.ticker = ticker
        self.file_path = RAW_DIR / f"{ticker}_history.csv"
        
    def add_indicators(self, df):
        df = df.copy()
        # 移动平均线
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 动量
        df['Momentum'] = df['close'] - df['close'].shift(10)
        
        # 波动率
        df['Volatility'] = df['close'].rolling(window=10).std()
        
        return df.dropna()

    def run_forecast(self):
        print(f"\n🔮 === 正在启动 {self.ticker} 水晶球预测程序 ===")
        
        if not os.path.exists(self.file_path):
            print(f"❌ 没找到数据文件: {self.file_path}")
            print("💡 请先运行 1. 数据采集 (Price Collector)")
            return
            
        df = pd.read_csv(self.file_path)

        # 强制小写列名
        df.columns = [c.lower() for c in df.columns]

        # 确保有 date 列
        if 'date' not in df.columns:
            print("❌ 数据格式错误：缺少 'date' 列")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        print(f"📅 数据最新日期: {df.index[-1].date()}")
        
        # 准备特征
        df_featured = self.add_indicators(df)
        
        if len(df_featured) < 50:
            print("❌ 数据量太少，无法进行技术预测")
            return

        # 目标：明天涨(1) 还是 跌(0)?
        df_featured['Target'] = (df_featured['close'].shift(-1) > df_featured['close']).astype(int)
        
        features = ['SMA_10', 'SMA_50', 'RSI', 'Momentum', 'Volatility']
        
        # 取最后一行作为“明天”的输入
        last_row = df_featured.iloc[[-1]][features]
        
        # 训练数据：去掉最后一行 (因为最后一行不知道明天涨跌，它是我们要预测的)
        train_data = df_featured.iloc[:-1]
        
        X = train_data[features]
        y = train_data['Target']
        
        # 训练
        print("🧠 正在学习所有历史走势 (技术面)...")
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        
        # 预测
        print("⚡️ 正在推演未来...")
        # prediction = model.predict(last_row)[0] # 暂时不用硬预测，只看概率
        probability = model.predict_proba(last_row)[0]
        
        # 输出
        print("\n" + "="*35)
        print(f"📢 {self.ticker} 下一个交易日预测 (纯技术面)")
        print("="*35)
        
        # probability[1] 是上涨的概率
        up_prob = probability[1] * 100
        
        if up_prob > 55:
            print(f"🚀 方向: 看涨 (BULLISH)")
            print(f"🔥 信心: {up_prob:.2f}%")
            print("💡 建议: 技术指标向好，多头排列。")
        elif up_prob < 45:
            print(f"📉 方向: 看跌 (BEARISH)")
            print(f"❄️ 信心: {(100-up_prob):.2f}%")
            print("💡 建议: 动能减弱或超买，注意风险。")
        else:
            print(f"🐢 方向: 震荡/看不清 (NEUTRAL)")
            print(f"⚖️ 信心: {up_prob:.2f}% (五五开)")
            print("💡 建议: 信号冲突，多看少动。")
            
        print("="*35 + "\n")
        
        curr_rsi = last_row['RSI'].values[0]
        curr_price = df.iloc[-1]['close']
        sma_50 = last_row['SMA_50'].values[0]
        
        print(f"📊 当前参考指标:")
        print(f" - 收盘价: ${curr_price:.2f}")
        print(f" - RSI (14): {curr_rsi:.2f}")
        print(f" - MA50生命线: ${sma_50:.2f}")
        
        if curr_price < sma_50:
            print("⚠️ 注意: 股价位于 50日均线 下方，长期趋势偏弱！")
        elif curr_price > sma_50:
             print("✅ 注意: 股价位于 50日均线 上方，长期趋势向好！")

# ==========================================
# 3. 🔌 统一接口 (供 main.py 调用)
# ==========================================
def run_tech_forecast(ticker="META"):
    forecaster = TechForecaster(ticker=ticker)
    forecaster.run_forecast()

if __name__ == "__main__":
    run_tech_forecast("META")