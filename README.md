# E-Commerce ETL & Customer Segmentation (PySpark)

An end-to-end data science project: a **PySpark ETL pipeline** that ingests messy raw e-commerce
data, cleans/validates it, loads a partitioned Parquet warehouse (+ SQLite mirror), and then uses
**Spark MLlib** to segment customers with KMeans.

```
 raw CSV / JSON            TRANSFORM                         LOAD                    ML
┌──────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐   ┌────────────────┐
│ orders.csv   │   │ safe type-casting        │   │ Parquet (snappy)    │   │ VectorAssembler│
│ customers.csv│──▶│ dedupe, standardise      │──▶│  fact_orders        │──▶│ StandardScaler │
│ products.json│   │ validate + quarantine    │   │   partitioned by    │   │ KMeans (k=4)   │
└──────────────┘   │ join + metrics           │   │   year_month        │   │ silhouette     │
   EXTRACT         │ window fns, RFM features │   │  dims, aggregates   │   │ segment naming │
 (explicit schemas)└──────────────────────────┘   │ SQLite warehouse.db │   └────────────────┘
                                                   └─────────────────────┘
```

## Quick start
```bash
pip install -r requirements.txt        # needs Java 17+ for Spark
python scripts/generate_data.py        # create messy synthetic data (40,800 order rows)
python src/pipeline.py                 # run Extract -> Transform -> Load -> ML  (~2 min locally)
python scripts/report.py               # charts in reports/
pytest -q                              # unit tests for the transform layer
```

## Project layout
| Path | Purpose |
|---|---|
| `scripts/generate_data.py` | Synthetic data with injected defects (duplicates, nulls, bad types, orphans, mixed date formats) |
| `src/etl/extract.py` | Reads sources with **explicit schemas** (orders as strings so bad values can be quarantined) |
| `src/etl/transform.py` | Pure, unit-testable functions: cleaning, validation, fact table, aggregates, RFM features |
| `src/etl/load.py` | Parquet writer (partitioning) and SQLite writer |
| `src/etl/quality.py` | Rejection summary, null report |
| `src/ml/segmentation.py` | MLlib pipeline, silhouette-based k search, behaviour-driven segment naming |
| `src/pipeline.py` | Orchestrator with logging and data-quality stats |
| `tests/test_transform.py` | pytest suite using a local SparkSession |

## Key engineering decisions
- **Quarantine, don't drop**: invalid orders go to `data/processed/rejected_records` with a `reject_reason`
  (missing/non-numeric quantity, non-positive quantity, missing price, orphan customer/product), so
  data loss is auditable.
- **ANSI-safe parsing**: Spark 4 runs in ANSI mode where a bad `CAST` throws. The pipeline uses
  `try_cast` / `try_to_timestamp` / `try_to_date`, which yield NULL, then validates explicitly.
- **Broadcast joins** for small dimension tables avoid shuffling the large fact table.
- **Window functions**: 7-day moving average of revenue (`rowsBetween`) and `lag()` to compute
  average days between a customer's orders.
- **Partitioning**: `fact_orders` is partitioned by `year_month` for partition pruning.
- **`cache()`** on reused DataFrames (valid orders, fact, features) to avoid recomputation.

## Results on the sample data
- 40,800 raw order rows -> 800 duplicates removed -> 37,988 valid, 2,012 quarantined.
- Segments (KMeans, k=4): *Champions* (~659 customers) generate ~94% of revenue; the remaining
  segments are lapsed/dormant customers, i.e. natural targets for win-back campaigns.
- Silhouette scores are printed for k=2..7. They favour a different k than the business-chosen k=4;
  `src/config.py` makes it a one-line change.

## Ideas to extend
- Swap the local source for S3/ADLS and `write.jdbc` into Postgres.
- Orchestrate with Airflow/Prefect; add Great Expectations checks.
- Add a churn classifier (`RandomForestClassifier`) using `customer_features`.
- Make it incremental: append new `year_month` partitions with `partitionOverwriteMode=dynamic`.
