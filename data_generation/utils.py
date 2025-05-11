from pathlib import Path
import pandas as pd

def deliver_original_data(assign_sid: bool = True):
    """
    Deliver the original Kaggle dataset
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    We assign `sid` to each original point to track lineage through
    oversampling methods. Downstream, `sid` will be shared between
    each original data point and the synthetic points that
    are generated from it.
    """
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "original_data.parquet"
    df = pd.read_parquet(data_path)
    if assign_sid and "sid" not in df.columns:
        df = df.reset_index(drop=True)
        df["sid"] = df.index.astype(str)
    return df
