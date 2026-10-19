"""LOAD: write curated data to the warehouse layer.

- Parquet (columnar, compressed, schema-embedded) is the primary lake format.
  The large fact table is partitioned by year_month so queries filtering on
  month only read the relevant folders (partition pruning).
- Small tables are also mirrored into SQLite to show a relational sink
  (for big data you'd use spark.write.jdbc with the proper driver instead).
"""
import sqlite3
from typing import Optional, Sequence

import pandas as pd
from pyspark.sql import DataFrame


def write_parquet(df: DataFrame, path: str, partition_by: Optional[Sequence[str]] = None,
                  mode: str = "overwrite") -> None:
    writer = df.write.mode(mode).option("compression", "snappy")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(path)


def write_sqlite(df: DataFrame, db_path: str, table: str) -> int:
    # collect() avoids the Arrow/pandas-version coupling of toPandas(); fine for small tables
    pdf = pd.DataFrame([r.asDict() for r in df.collect()], columns=df.columns)
    for c in pdf.columns:                      # SQLite has no native date/bool types
        if str(pdf[c].dtype).startswith(("datetime", "object")) and c.endswith(("date", "ts")):
            pdf[c] = pdf[c].astype(str)
    with sqlite3.connect(db_path) as con:
        pdf.to_sql(table, con, if_exists="replace", index=False)
    return len(pdf)
