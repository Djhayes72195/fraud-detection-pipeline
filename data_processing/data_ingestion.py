import pandas as pd
from .config import S3_BUCKET, S3_PREFIX


def load_days(start_day: int, end_day: int):
    dfs = []
    for day in range(start_day, end_day + 1):
        date_str = f"2021-01-{day:02d}"
        path = f"s3://{S3_BUCKET}/{S3_PREFIX}/dt={date_str}/transactions.parquet"
        df_day = pd.read_parquet(
            path, storage_options={"anon": False}
        )  # storage_options arg --> don't assume public, use AWS credentials
        dfs.append(df_day)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all
