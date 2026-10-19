"""Data-quality reporting."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def rejection_summary(rejected: DataFrame) -> DataFrame:
    return rejected.groupBy("reject_reason").count().orderBy(F.desc("count"))


def null_report(df: DataFrame) -> dict:
    row = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first()
    return {k: v for k, v in row.asDict().items() if v}
