from pathlib import Path

LOOKBACK_WINDOW = 100
DEFAULT_START_DATE = "2015-01-01"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "keras_model.h5"
YFINANCE_CACHE_DIR = PROJECT_ROOT / ".cache"
