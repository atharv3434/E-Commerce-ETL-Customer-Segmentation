"""Central configuration for paths and pipeline parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_ORDERS = str(RAW_DIR / "orders.csv")
RAW_CUSTOMERS = str(RAW_DIR / "customers.csv")
RAW_PRODUCTS = str(RAW_DIR / "products.json")

OUT_FACT_ORDERS = str(PROCESSED_DIR / "fact_orders")
OUT_DIM_CUSTOMERS = str(PROCESSED_DIR / "dim_customers")
OUT_DIM_PRODUCTS = str(PROCESSED_DIR / "dim_products")
OUT_CUSTOMER_FEATURES = str(PROCESSED_DIR / "customer_features")
OUT_DAILY_SALES = str(PROCESSED_DIR / "agg_daily_sales")
OUT_SEGMENTS = str(PROCESSED_DIR / "customer_segments")
OUT_REJECTS = str(PROCESSED_DIR / "rejected_records")
SQLITE_DB = str(PROCESSED_DIR / "warehouse.db")

N_CLUSTERS = 4
RANDOM_SEED = 42
