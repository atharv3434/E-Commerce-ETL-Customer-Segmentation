"""SparkSession factory."""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "ecommerce-etl", master: str = "local[*]") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")  # small data -> fewer shuffle partitions
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
