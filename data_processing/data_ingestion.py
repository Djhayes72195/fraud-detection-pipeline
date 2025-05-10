from pyspark.sql import SparkSession
from .config import S3_BUCKET, S3_PREFIX

def get_spark_session(app_name="Fraud Detection Training"):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_days(start_day: int, end_day: int):
    spark = get_spark_session()
    dfs = []
    for day in range(start_day, end_day + 1):
        path = f"s3://{S3_BUCKET}/{S3_PREFIX}/dt=2021-01-{day:02d}/"
        dfs.append(spark.read.parquet(path))
    df_all = dfs[0]
    for df in dfs[1:]:
        df_all = df_all.union(df)
    return df_all
