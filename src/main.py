import sys
import os
import time
from pathlib import Path
from src.config import DEFAULT_TICKER
# ==========================================
# 1. 🟢 路径配置
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

print(f"📂 项目根目录: {project_root}")

# ==========================================
# 2. 🔌 动态加载模块
# ==========================================
modules = {}

# --- 1. 采集模块 ---
try:
    from src.collector.news_collector import run_news_collector
    modules['news_collector'] = run_news_collector
except ImportError: pass

try:
    from src.collector.price_collector import run_price_collector
    modules['price_collector'] = run_price_collector
except ImportError: pass

# --- 2. 分析模块 ---
try:
    from src.analysis.analyzer import run_sentiment_analysis
    modules['analyzer'] = run_sentiment_analysis
except ImportError: pass

# --- 3. 训练模块 (Sentiment Trend Pro) ---
try:
    from src.model.sentiment_trend_pro import run_train_model
    modules['trainer'] = run_train_model
except ImportError: pass

# --- 4. 回测模块 ---
try:
    from src.strategy.backtester1 import backtester1
    modules['backtester'] = backtester1
except ImportError: pass

# --- 5. 预测模块 (Tech Forecaster - 新增!) ---
try:
    from src.model.forecast_tech import run_tech_forecast
    modules['forecaster'] = run_tech_forecast
    print("✅ 模块加载成功: 技术面水晶球 (Forecaster)")
except ImportError as e:
    print(f"⚠️ 模块未加载: 技术面预测 ({e})")


# ==========================================
# 3. 🎮 主控制台
# ==========================================
def main():
    default_ticker = DEFAULT_TICKER
    
    while True:
        print("\n" + "="*50)
        print(f"   🤖 AI 量化交易系统 | 当前目标: {default_ticker}")
        print("="*50)
        
        print("1. 📥  数据采集 (Step 1: Get Data)") 
        print("2. 🧠  情绪分析 (Step 2: Analysis)")
        print("3. 🏋️  AI模型训练 (Step 3: Training)")
        print("4. 📈  策略回测 (Step 4: Backtest)")
        print("-" * 30)
        print("5. 🔮  【明日预测】 (纯技术面水晶球)")
        print("-" * 30)
        print("9. 🚀  【全流程自动化】 (1 -> 2 -> 3 -> 4)")
        print("-" * 30)
        print("C. 🔄  更改股票代码")
        print("0. 🚪  退出系统")
        
        choice = input("\n👉 请输入指令: ").upper().strip()

        # --- 1. 采集 ---
        if choice == "1":
            print("\n>>> 启动数据采集...")
            if 'news_collector' in modules: modules['news_collector'](default_ticker)
            if 'price_collector' in modules: modules['price_collector'](default_ticker)

        # --- 2. 分析 ---
        elif choice == "2":
            print("\n>>> 启动情绪分析...")
            if 'analyzer' in modules: modules['analyzer'](default_ticker)

        # --- 3. 训练 ---
        elif choice == "3":
            print("\n>>> 启动 AI 模型训练...")
            if 'trainer' in modules:
                modules['trainer'](default_ticker)
            else:
                print("❌ 错误: 训练模块 (sentiment_trend_pro) 未找到")

        # --- 4. 回测 ---
        elif choice == "4":
            print("\n>>> 启动历史回测...")
            if 'backtester' in modules: modules['backtester'](default_ticker)

        # --- 5. 预测 (新功能) ---
        elif choice == "5":
            print(f"\n>>> 启动 {default_ticker} 明日走势预测...")
            if 'forecaster' in modules:
                modules['forecaster'](default_ticker)
            else:
                print("❌ 错误: 预测模块 (forecast_tech) 未找到")

        # --- 9. 全自动 ---
        elif choice == "9":
            print(f"\n🚀 启动全流程自动化: {default_ticker}")
            
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
                print("\n[Step 3] 训练 AI 模型...")
                modules['trainer'](default_ticker)
                time.sleep(1)

            # Step 4
            if 'backtester' in modules:
                print("\n[Step 4] 执行回测验证...")
                modules['backtester'](default_ticker)
            
            # 额外赠送：顺便跑一下明日预测
            if 'forecaster' in modules:
                 print("\n[Bonus] 生成明日预测报告...")
                 modules['forecaster'](default_ticker)
            
            print("\n🎉 全流程执行完毕！")

        # --- 其他 ---
        elif choice == "C":
            new_ticker = input("请输入新代码 (例如 NVDA): ").upper()
            if new_ticker: default_ticker = new_ticker
        elif choice == "0":
            print("\n👋 祝交易顺利，赚大钱！")
            break
        else:
            print("\n❌ 无效指令")

if __name__ == "__main__":
    main()