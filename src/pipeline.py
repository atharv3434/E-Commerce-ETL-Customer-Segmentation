"""End-to-end orchestration: Extract -> Transform -> Load -> ML."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F  # noqa: E402

from src import config as C  # noqa: E402
from src.etl import extract, load, quality, transform  # noqa: E402
from src.ml import segmentation as seg  # noqa: E402
from src.spark_session import get_spark  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")


def main() -> None:
    t0 = time.time()
    spark = get_spark()

    # ---------------------------- EXTRACT ----------------------------
    log.info("EXTRACT  reading raw sources")
    raw_orders = extract.extract_orders(spark, C.RAW_ORDERS)
    raw_customers = extract.extract_customers(spark, C.RAW_CUSTOMERS)
    raw_products = extract.extract_products(spark, C.RAW_PRODUCTS)
    n_raw = raw_orders.count()
    log.info(f"         raw orders={n_raw:,} customers={raw_customers.count():,} products={raw_products.count()}")

    # --------------------------- TRANSFORM ---------------------------
    log.info("TRANSFORM cleaning & validating")
    customers = transform.clean_customers(raw_customers).cache()
    products = transform.clean_products(raw_products).cache()
    parsed = transform.parse_orders(raw_orders)
    valid, rejected = transform.validate_orders(parsed, customers, products)
    valid, rejected = valid.cache(), rejected.cache()

    n_parsed, n_valid, n_rej = parsed.count(), valid.count(), rejected.count()
    log.info(f"         duplicates removed={n_raw - n_parsed:,} | valid={n_valid:,} | rejected={n_rej:,}")
    log.info("         rejection breakdown:")
    for r in quality.rejection_summary(rejected).collect():
        log.info(f"            {r['reject_reason']:<35}{r['count']:>6,}")
    bad_emails = customers.filter(~F.col("is_valid_email")).count()
    log.info(f"         customers with invalid email (nulled)={bad_emails}")

    fact = transform.build_fact_orders(valid, products).cache()
    daily = transform.daily_sales(fact)
    features = transform.customer_features(fact, customers).cache()
    log.info(f"         customer feature rows={features.count():,}")

    # ------------------------------ LOAD ------------------------------
    log.info("LOAD     writing parquet + sqlite")
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load.write_parquet(fact, C.OUT_FACT_ORDERS, partition_by=["year_month"])
    load.write_parquet(customers, C.OUT_DIM_CUSTOMERS)
    load.write_parquet(products, C.OUT_DIM_PRODUCTS)
    load.write_parquet(daily, C.OUT_DAILY_SALES)
    load.write_parquet(features, C.OUT_CUSTOMER_FEATURES)
    load.write_parquet(rejected, C.OUT_REJECTS)
    for name, df in [("dim_customers", customers), ("dim_products", products),
                     ("agg_daily_sales", daily), ("customer_features", features)]:
        log.info(f"         sqlite.{name}: {load.write_sqlite(df, C.SQLITE_DB, name):,} rows")

    # ------------------------------- ML -------------------------------
    log.info("ML       customer segmentation (KMeans)")
    best_k, scores = seg.select_k(features, seed=C.RANDOM_SEED)
    log.info("         silhouette by k: " + ", ".join(f"k={k}:{s:.3f}" for k, s in scores.items()))
    k = C.N_CLUSTERS  # business-driven k; silhouette shown for reference
    log.info(f"         best silhouette k={best_k}; using business k={k}")
    _, scored = seg.fit_segments(features, k, seed=C.RANDOM_SEED)
    labeled = seg.label_segments(scored).cache()
    load.write_parquet(labeled, C.OUT_SEGMENTS)
    load.write_sqlite(labeled, C.SQLITE_DB, "customer_segments")

    print("\n=== SEGMENT PROFILE ===")
    seg.segment_profile(labeled).show(truncate=False)
    print("=== TOP CATEGORIES BY NET REVENUE ===")
    (fact.filter(F.col("status") == "completed").groupBy("category")
         .agg(F.round(F.sum("net_revenue"), 0).alias("revenue"), F.round(F.sum("profit"), 0).alias("profit"))
         .orderBy(F.desc("revenue")).show())

    log.info(f"DONE in {time.time() - t0:.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
