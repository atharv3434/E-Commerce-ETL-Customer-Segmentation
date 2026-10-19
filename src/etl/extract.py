"""
EXTRACT: read raw sources with explicit schemas (no schema inference).

Orders are read with every column as STRING on purpose: the raw feed contains
values like "two" in numeric columns, and we want to parse them ourselves so
bad rows can be quarantined rather than silently turned into nulls or crashing the job.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (DoubleType, StringType, StructField, StructType)

ORDERS_SCHEMA = StructType([StructField(c, StringType(), True) for c in
    ["order_id", "customer_id", "product_id", "quantity", "unit_price",
     "discount_pct", "order_ts", "status"]])

CUSTOMERS_SCHEMA = StructType([StructField(c, StringType(), True) for c in
    ["customer_id", "name", "email", "country", "signup_date"]])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("unit_cost", DoubleType(), True),
    StructField("list_price", DoubleType(), True),
])


def extract_orders(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, schema=ORDERS_SCHEMA)


def extract_customers(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, schema=CUSTOMERS_SCHEMA)


def extract_products(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.schema(PRODUCTS_SCHEMA).json(path)
