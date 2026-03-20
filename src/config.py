import os
from pathlib import Path

# ==========================================
# 1. Project root anchor
# ==========================================
# This keeps path resolution stable regardless of where the script is launched.
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 2. Global defaults
# ==========================================
# Change the default research ticker here, for example: TSLA, NVDA, AAPL.
DEFAULT_TICKER = "NVDA"

# ==========================================
# 3. Core directory paths
# ==========================================
DATA_DIR = BASE_DIR / "data"

# Raw source data such as news articles and downloaded price history.
RAW_DIR = DATA_DIR / "raw"

# Processed datasets such as daily sentiment scores and merged features.
PROCESSED_DIR = DATA_DIR / "processed"

# Saved model artifacts.
MODEL_DIR = BASE_DIR / "models"

# Sentiment lexicon resources.
LEXICON_DIR = DATA_DIR / "lexicon"

# ==========================================
# 4. Directory initialization
# ==========================================
# Create required directories on startup if they do not exist.
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, LEXICON_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 5. Shared path references
# ==========================================
# The trainer uses this as the base output location for model files.
MODEL_PATH = MODEL_DIR

# Full path to the sentiment lexicon file if needed by analyzers.
LEXICON_PATH = LEXICON_DIR / "LoughranMcDonald_MasterDictionary.csv"

# ==========================================
# Deprecated examples
# ==========================================
# Older versions hard-coded paths such as "AAPL_history.csv".
# The current structure builds file names dynamically from the ticker,
# so the system can support multiple symbols without code changes.
#
# STOCK_HISTORY_PATH = RAW_DIR / "AAPL_history.csv"   # Avoid this pattern
# FINAL_MERGED_PATH = PROCESSED_DIR / "AAPL_merged.csv"  # Avoid this pattern
