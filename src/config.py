import os
from pathlib import Path

# ==========================================
# 1. 📍 项目根目录锚点
# ==========================================
# 无论你在哪里运行脚本，这里都能找到项目的绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 2. ⚙️ 全局默认设置 (总开关)
# ==========================================
# 想换股研究？直接改这里！(例如 "TSLA", "NVDA", "AAPL")
DEFAULT_TICKER = "META"

# ==========================================
# 3. 📂 核心文件夹路径
# ==========================================
DATA_DIR = BASE_DIR / "data"

# 存放原始数据 (爬虫下来的新闻、yfinance下载的股价)
RAW_DIR = DATA_DIR / "raw"

# 存放处理后的数据 (计算好情绪分数的表、合并了技术指标的表)
PROCESSED_DIR = DATA_DIR / "processed"

# 存放训练好的 AI 模型 (.pkl 文件)
MODEL_DIR = BASE_DIR / "models"

# 存放情感词典 (Loughran McDonald 等)
LEXICON_DIR = DATA_DIR / "lexicon"

# ==========================================
# 4. 🛠 自动初始化目录
# ==========================================
# 如果文件夹不存在，自动创建它们，防止报错
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, LEXICON_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 5. 🔗 通用文件路径引用
# ==========================================
# 训练模块会引用这个变量。
# 设置为目录，让 Trainer 自动生成类似 models/META_model.pkl 的文件名
MODEL_PATH = MODEL_DIR

# 情感词典的具体路径 (如果你的分析器需要用到)
LEXICON_PATH = LEXICON_DIR / "LoughranMcDonald_MasterDictionary.csv"

# ==========================================
# ⚠️ 已弃用 (Deprecated) - 不要取消注释
# ==========================================
# 之前的版本这里写死了 "AAPL_history.csv" 等路径。
# 在新版架构中，文件名由各个模块根据 `ticker` 变量动态生成。
# 这样你的系统才能同时支持 META, TSLA, NVDA 等多只股票，而不需要改代码。
#
# STOCK_HISTORY_PATH = RAW_DIR / "AAPL_history.csv"  <-- ❌ 不要用这种写法了
# FINAL_MERGED_PATH = PROCESSED_DIR / "AAPL_merged.csv" <-- ❌ 也不要用这种