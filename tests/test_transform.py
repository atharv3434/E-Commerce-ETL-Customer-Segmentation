"""Unit tests for the transform layer (run with: pytest -q)."""
import pytest
from pyspark.sql import SparkSession

from src.etl import extract, transform


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def _orders(spark, rows):
    cols = ["order_id", "customer_id", "product_id", "quantity", "unit_price", "discount_pct", "order_ts", "status"]
    return spark.createDataFrame(rows, schema=extract.ORDERS_SCHEMA).select(cols)


def test_parse_orders_handles_bad_values_and_dedupes(spark):
    df = _orders(spark, [
        ("O1", "C1", "P1", "2", "10.0", "0", "2024-01-05 10:00:00", " COMPLETED "),
        ("O1", "C1", "P1", "2", "10.0", "0", "2024-01-05 10:00:00", " COMPLETED "),  # duplicate
        ("O2", "C1", "P1", "two", "10.0", "0", "05/01/2024 10:00", "returned"),       # bad qty, alt date fmt
    ])
    out = {r["order_id"]: r for r in transform.parse_orders(df).collect()}
    assert len(out) == 2
    assert out["O1"]["status"] == "completed"
    assert out["O2"]["quantity"] is None
    assert out["O2"]["order_ts"] is not None   # dd/MM/yyyy format parsed


def test_validate_orders_splits_and_labels_rejects(spark):
    cust = spark.createDataFrame([("C1",)], ["customer_id"])
    prod = spark.createDataFrame([("P1",)], ["product_id"])
    df = transform.parse_orders(_orders(spark, [
        ("O1", "C1", "P1", "1", "5", "0", "2024-01-01 00:00:00", "completed"),
        ("O2", "C1", "P1", "-1", "5", "0", "2024-01-01 00:00:00", "completed"),
        ("O3", "C9", "P1", "1", "5", "0", "2024-01-01 00:00:00", "completed"),
        ("O4", "C1", "P9", "1", "5", "0", "2024-01-01 00:00:00", "completed"),
        ("O5", "C1", "P1", "1", None, "0", "2024-01-01 00:00:00", "completed"),
    ]))
    valid, rejected = transform.validate_orders(df, cust, prod)
    assert [r.order_id for r in valid.collect()] == ["O1"]
    reasons = {r.order_id: r.reject_reason for r in rejected.collect()}
    assert reasons == {"O2": "quantity_not_positive", "O3": "orphan_customer",
                       "O4": "orphan_product", "O5": "price_missing"}


def test_clean_customers_standardises_country_and_email(spark):
    df = spark.createDataFrame(
        [("C1", " A ", "a@x.com", " india ", "2023-01-01"),
         ("C2", "B", "bad_email", "United States", "2023-02-01"),
         ("C2", "B", "bad_email", "United States", "2023-02-01")],
        schema=extract.CUSTOMERS_SCHEMA)
    out = {r.customer_id: r for r in transform.clean_customers(df).collect()}
    assert len(out) == 2
    assert out["C1"].country == "India" and out["C1"].name == "A"
    assert out["C2"].country == "USA" and out["C2"].email is None


def test_fact_revenue_and_profit(spark):
    valid = transform.parse_orders(_orders(spark, [("O1", "C1", "P1", "2", "100", "10", "2024-03-02 09:00:00", "completed")]))
    prod = spark.createDataFrame([("P1", "Electronics", 60.0)], ["product_id", "category", "unit_cost"])
    r = transform.build_fact_orders(valid, prod).first()
    assert r.gross_revenue == 200.0
    assert r.net_revenue == 180.0            # 10% discount
    assert r.profit == 60.0                  # 180 - 2*60
    assert r.year_month == "2024-03" and r.is_weekend is True   # 2 Mar 2024 = Saturday
