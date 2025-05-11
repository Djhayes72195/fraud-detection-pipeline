from pathlib import Path
import pandas as pd

def deliver_original_data(assign_uid: bool = True):
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "original_data.parquet"
    df = pd.read_parquet(data_path)
    if assign_uid and "uid" not in df.columns:
        df = df.reset_index(drop=True)
        df["uid"] = df.index.astype(str)
    return df
