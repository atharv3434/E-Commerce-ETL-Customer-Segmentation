"""
TRANSFORM: cleaning, validation, enrichment and feature engineering.

Every function is pure (DataFrame in -> DataFrame out) so it can be unit-tested.
Note: Spark 4 runs in ANSI mode by default, where a bad CAST raises an error.
We therefore use try_cast / try_to_timestamp, which return NULL on failure.
"""
from typing import Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

COUNTRY_MAP = {
    "india": "India", "in": "India",
    "usa": "USA", "us": "USA", "united states": "USA",
    "uk": "UK", "united kingdom": "UK",
    "germany": "Germany", "de": "Germany",
    "france": "France", "fr": "France",
    "canada": "Canada", "ca": "Canada",
}
EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


# --------------------------------------------------------------------------
# Dimension tables
# --------------------------------------------------------------------------
def clean_customers(df: DataFrame) -> DataFrame:
    mapping = F.create_map(*[x for kv in COUNTRY_MAP.items() for x in (F.lit(kv[0]), F.lit(kv[1]))])
    return (
        df.dropDuplicates(["customer_id"])
        .withColumn("name", F.trim("name"))
        .withColumn("country_key", F.lower(F.trim("country")))
        .withColumn("country", F.coalesce(mapping[F.col("country_key")], F.initcap("country_key")))
        .drop("country_key")
        .withColumn("signup_date", F.expr("try_to_date(signup_date, 'yyyy-MM-dd')"))
        .withColumn("is_valid_email", F.col("email").rlike(EMAIL_REGEX))
        .withColumn("email", F.when(F.col("is_valid_email"), F.lower("email")))  # null out bad emails
    )


def clean_products(df: DataFrame) -> DataFrame:
    return (
        df.dropDuplicates(["product_id"])
        .withColumn("category", F.initcap(F.trim("category")))
        .withColumn("margin_pct", F.round((F.col("list_price") - F.col("unit_cost")) / F.col("list_price") * 100, 2))
    )


# --------------------------------------------------------------------------
# Orders: parse -> validate -> split good / rejected
# --------------------------------------------------------------------------
def parse_orders(df: DataFrame) -> DataFrame:
    """Type-cast raw strings safely and standardise text columns."""
    ts = F.coalesce(
        F.expr("try_to_timestamp(order_ts, 'yyyy-MM-dd HH:mm:ss')"),
        F.expr("try_to_timestamp(order_ts, 'dd/MM/yyyy HH:mm')"),
    )
    return (
        df.dropDuplicates()                                   # exact duplicate rows
        .dropDuplicates(["order_id"])                         # same order id, keep one
        .withColumn("quantity", F.expr("try_cast(quantity as int)"))
        .withColumn("unit_price", F.expr("try_cast(unit_price as double)"))
        .withColumn("discount_pct", F.expr("try_cast(discount_pct as double)"))
        .withColumn("order_ts", ts)
        .withColumn("status", F.lower(F.trim("status")))
    )


def validate_orders(df: DataFrame, customers: DataFrame, products: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Return (valid_orders, rejected_orders). Each rejected row carries a reject_reason."""
    cust_ids = customers.select("customer_id").withColumn("_c", F.lit(True))
    prod_ids = products.select("product_id").withColumn("_p", F.lit(True))

    flagged = (
        df.join(F.broadcast(cust_ids), "customer_id", "left")
          .join(F.broadcast(prod_ids), "product_id", "left")
          .withColumn(
              "reject_reason",
              F.when(F.col("quantity").isNull(), "quantity_missing_or_not_numeric")
               .when(F.col("quantity") <= 0, "quantity_not_positive")
               .when(F.col("unit_price").isNull(), "price_missing")
               .when(F.col("order_ts").isNull(), "timestamp_unparseable")
               .when(F.col("_c").isNull(), "orphan_customer")
               .when(F.col("_p").isNull(), "orphan_product"),
          )
          .drop("_c", "_p")
    )
    return flagged.filter(F.col("reject_reason").isNull()).drop("reject_reason"), \
           flagged.filter(F.col("reject_reason").isNotNull())


def build_fact_orders(valid: DataFrame, products: DataFrame) -> DataFrame:
    """Enrich orders with product info and compute financial metrics."""
    return (
        valid.join(F.broadcast(products.select("product_id", "category", "unit_cost")), "product_id", "left")
        .withColumn("gross_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
        .withColumn("net_revenue", F.round(F.col("gross_revenue") * (1 - F.col("discount_pct") / 100), 2))
        .withColumn("profit", F.round(F.col("net_revenue") - F.col("quantity") * F.col("unit_cost"), 2))
        .withColumn("order_date", F.to_date("order_ts"))
        .withColumn("year_month", F.date_format("order_ts", "yyyy-MM"))
        .withColumn("hour_of_day", F.hour("order_ts"))
        .withColumn("is_weekend", F.dayofweek("order_ts").isin(1, 7))
        .drop("unit_cost")
    )


# --------------------------------------------------------------------------
# Aggregates & features
# --------------------------------------------------------------------------
def daily_sales(fact: DataFrame) -> DataFrame:
    """Daily revenue by category with a 7-day moving average (window function)."""
    base = (
        fact.filter(F.col("status") == "completed")
        .groupBy("order_date", "category")
        .agg(F.sum("net_revenue").alias("revenue"),
             F.countDistinct("order_id").alias("orders"),
             F.sum("profit").alias("profit"))
    )
    w = Window.partitionBy("category").orderBy("order_date").rowsBetween(-6, 0)
    return base.withColumn("revenue_7d_avg", F.round(F.avg("revenue").over(w), 2))


def customer_features(fact: DataFrame, customers: DataFrame) -> DataFrame:
    """RFM + behavioural features per customer (the ML input table)."""
    completed = fact.filter(F.col("status") == "completed")
    snapshot = fact.agg(F.max("order_date")).first()[0]

    # average gap between consecutive orders, via LAG
    w = Window.partitionBy("customer_id").orderBy("order_ts")
    gaps = (
        completed.withColumn("prev_ts", F.lag("order_ts").over(w))
        .withColumn("gap_days", (F.unix_timestamp("order_ts") - F.unix_timestamp("prev_ts")) / 86400)
        .groupBy("customer_id").agg(F.avg("gap_days").alias("avg_days_between_orders"))
    )

    status_stats = fact.groupBy("customer_id").agg(
        F.avg(F.when(F.col("status") == "returned", 1.0).otherwise(0.0)).alias("return_rate"))

    agg = completed.groupBy("customer_id").agg(
        F.max("order_date").alias("last_order_date"),
        F.countDistinct("order_id").alias("frequency"),
        F.sum("net_revenue").alias("monetary"),
        F.avg("net_revenue").alias("avg_order_value"),
        F.avg("discount_pct").alias("avg_discount_pct"),
        F.countDistinct("category").alias("distinct_categories"),
        F.avg(F.col("is_weekend").cast("int")).alias("weekend_share"),
    )

    return (
        agg.join(gaps, "customer_id", "left").join(status_stats, "customer_id", "left")
        .join(customers.select("customer_id", "country", "signup_date"), "customer_id", "left")
        .withColumn("recency_days", F.datediff(F.lit(snapshot), "last_order_date"))
        .withColumn("tenure_days", F.datediff(F.lit(snapshot), "signup_date"))
        .fillna({"avg_days_between_orders": 0.0, "return_rate": 0.0})
        .withColumn("monetary", F.round("monetary", 2))
        .withColumn("avg_order_value", F.round("avg_order_value", 2))
        .select("customer_id", "country", "recency_days", "frequency", "monetary",
                "avg_order_value", "avg_discount_pct", "distinct_categories",
                "weekend_share", "avg_days_between_orders", "return_rate", "tenure_days")
    )
