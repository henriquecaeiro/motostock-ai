"""Application paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "xgboost_model.pkl"
MODELING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
DAILY_SALES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_product_sales.csv"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

SELECTED_MODEL_NAME = "xgboost"
DEFAULT_LEAD_TIME_DAYS = 7

VALID_STOCK_STATUSES = ("critical", "warning", "healthy", "overstock")
