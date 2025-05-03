from pathlib import Path
import pandas as pd

def deliver_original_data():
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "original_data.parquet"
    df = pd.read_parquet(data_path)
    return df
