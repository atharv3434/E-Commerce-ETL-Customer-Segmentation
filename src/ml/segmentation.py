"""Customer segmentation with Spark MLlib (StandardScaler + KMeans)."""
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

FEATURES: List[str] = ["recency_days", "frequency", "monetary", "avg_order_value",
                       "avg_discount_pct", "distinct_categories", "avg_days_between_orders"]


def _pipeline(k: int, seed: int) -> Pipeline:
    return Pipeline(stages=[
        VectorAssembler(inputCols=FEATURES, outputCol="raw_features", handleInvalid="skip"),
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
        KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster"),
    ])


def select_k(df: DataFrame, k_range=range(2, 8), seed: int = 42) -> Tuple[int, Dict[int, float]]:
    """Pick k with the best silhouette score."""
    scores: Dict[int, float] = {}
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                    metricName="silhouette", distanceMeasure="squaredEuclidean")
    for k in k_range:
        scores[k] = evaluator.evaluate(_pipeline(k, seed).fit(df).transform(df))
    return max(scores, key=scores.get), scores


def fit_segments(df: DataFrame, k: int, seed: int = 42) -> Tuple[PipelineModel, DataFrame]:
    model = _pipeline(k, seed).fit(df)
    return model, model.transform(df).drop("raw_features", "features")


def label_segments(scored: DataFrame, lapsed_after_days: int = 120, dormant_after_days: int = 300) -> DataFrame:
    """Name clusters from their measured behaviour (not from an arbitrary rank).

    - the highest-revenue cluster is "Champions"
    - every other cluster is "<Active|Lapsed|Dormant> <Higher|Lower>-Value", based on average
      recency (<= lapsed_after_days, <= dormant_after_days, beyond) and on revenue relative
      to the median cluster.
    """
    rows = (scored.groupBy("cluster")
            .agg(F.avg("monetary").alias("m"), F.avg("recency_days").alias("r"))
            .orderBy(F.desc("m")).collect())
    median_m = sorted(r["m"] for r in rows)[len(rows) // 2]
    mapping, used = {}, set()
    for i, r in enumerate(rows):
        if i == 0:
            name = "Champions"
        else:
            state = ("Active" if r["r"] <= lapsed_after_days
                     else "Lapsed" if r["r"] <= dormant_after_days else "Dormant")
            name = (f"{state} "
                    f"{'Higher' if r['m'] >= median_m else 'Lower'}-Value")
        while name in used:                      # guarantee unique names
            name += " (2)"
        used.add(name)
        mapping[r["cluster"]] = name
    expr = F.create_map(*[x for k, v in mapping.items() for x in (F.lit(k), F.lit(v))])
    return scored.withColumn("segment", expr[F.col("cluster")])


def segment_profile(labeled: DataFrame) -> DataFrame:
    return (labeled.groupBy("segment").agg(
        F.count("*").alias("customers"),
        F.round(F.avg("recency_days"), 1).alias("avg_recency_days"),
        F.round(F.avg("frequency"), 1).alias("avg_orders"),
        F.round(F.avg("monetary"), 0).alias("avg_revenue"),
        F.round(F.sum("monetary"), 0).alias("total_revenue"),
    ).orderBy(F.desc("avg_revenue")))
