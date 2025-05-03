import pandas as pd
from pathlib import Path
from datetime import timedelta, date
import numpy as np
from utils import deliver_original_data

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "raw_data"

def simulate_30_day_dataset(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Expands a base dataset into 30 days of synthetic data with light drift.

    Returns:
        Full concatenated DataFrame (30 days of data)
    """
    V_cols = [
        x for x in df_base.columns if x not in [
            "Time",
            "Amount",
            "Class"
        ]
    ]

    for day in range(30):
        if day not in [5, 15, 25]:  # For local testing
            continue
        df_day = df_base.copy()
        df_day["Day"] = day

        # Original dataset has two days. I want to simulate a single day per file
        df_day["Time"] = df_day["Time"] % 86400  # 86400 seconds in a day

        if day < 10:
            df_day = first_10_perturbation(df_day, V_cols)

        elif day >= 10 and day < 20:
            df_day = middle_10_perturbation(df_day, V_cols)

        else: # day >= 20
            df_day = last_10_perturbation(df_day, V_cols)

        # Replace with S3 integration when we scale
        write_sim_day(df_day, day)

def write_sim_day(df_day, day):
    batch_date = date(2021, 1, 1) + timedelta(days=day)
    out_path = OUTPUT_DIR / str(batch_date)
    out_path.mkdir(parents=True, exist_ok=True)

    df_day.to_parquet(out_path / "transactions.parquet", index=False)
    print(f"Saved {len(df_day)} rows to {out_path / 'transactions.parquet'}")

def first_10_perturbation(df_day, V_cols):
    """
    Apply light Gaussian noise to features for Days 0-9.
    Intended to simulate low-level randomness without introducing drift.
    """
    for col in V_cols:
        df_day[col] += np.random.normal(0, 0.02, size=len(df_day))

    df_day["Time"] += np.random.normal(0, 300, size=len(df_day))  # ±5 minutes
    df_day["Time"] = df_day["Time"].clip(lower=0, upper=86399)

    df_day["Amount"] += np.random.normal(loc=0, scale=5, size=len(df_day))
    df_day["Amount"] = df_day["Amount"].clip(lower=0.01)  # NOTE: clip could lead to unnatural clustering at 1 cent. Revisit if needed.
    
    df_day["PerturbationScheme"] = 1
    return df_day

def middle_10_perturbation(df_day, V_cols):
    """
    Apply covariate drift to simulate feature distribution shift during Days 10-19.
    Adds bias and variance to select V columns, and increases noise in Amount and Time.
    """
    # Slight drift in V4–V8: additive bias + increased variance
    drift_cols = [col for col in V_cols if col in ["V4", "V5", "V6", "V7", "V8"]]
    for col in drift_cols:
        df_day[col] += 0.2 + np.random.normal(0, 0.05, size=len(df_day))  # shift + jitter

    # Increased noise in Time: ±10 min
    df_day["Time"] += np.random.normal(0, 600, size=len(df_day))
    df_day["Time"] = df_day["Time"].clip(lower=0, upper=86399)

    # Add higher variance to Amount
    df_day["Amount"] += np.random.normal(loc=0, scale=10, size=len(df_day))
    df_day["Amount"] = df_day["Amount"].clip(lower=0.01)  # NOTE: clip could lead to unnatural clustering at 1 cent. Revisit if needed.

    df_day["PerturbationScheme"] = 2
    return df_day


def last_10_perturbation(df_day, V_cols):
    """
    Apply concept drift by modifying fraud patterns in Days 20–29.
    Amplifies fraud volume and shifts its feature distributions,
    while still perturbing non-fraud to avoid trivial separation.
    """
    fraud = df_day[df_day["Class"] == 1].copy()
    nonfraud = df_day[df_day["Class"] == 0].copy()

    # Perturb non-fraud lightly (same as first_10_perturbation)
    for col in V_cols:
        nonfraud[col] += np.random.normal(0, 0.02, size=len(nonfraud))
    nonfraud["Time"] += np.random.normal(0, 300, size=len(nonfraud))
    nonfraud["Time"] = nonfraud["Time"].clip(lower=0, upper=86399)
    nonfraud["Amount"] += np.random.normal(loc=0, scale=5, size=len(nonfraud))
    nonfraud["Amount"] = nonfraud["Amount"].clip(lower=0.01)

    # Generate bot fraud
    bot_fraud = fraud.sample(frac=2.0, replace=True).copy()
    drift_cols = [col for col in V_cols if col in ["V10", "V11", "V12", "V13"]]
    for col in drift_cols:
        bot_fraud[col] += np.random.normal(loc=1.0, scale=0.1, size=len(bot_fraud))
    bot_fraud["Time"] = np.random.normal(loc=2 * 3600, scale=1800, size=len(bot_fraud))
    bot_fraud["Time"] = bot_fraud["Time"].clip(0, 86399)
    bot_fraud["Amount"] += np.random.normal(loc=0, scale=15, size=len(bot_fraud))
    bot_fraud["Amount"] = bot_fraud["Amount"].clip(lower=0.01)

    # Merge and mark
    df_day = pd.concat([nonfraud, fraud, bot_fraud], ignore_index=True)
    df_day["PerturbationScheme"] = 3
    return df_day

if __name__ == "__main__":
    df_base = deliver_original_data()  # Loads and normalizes timestamps inside
    simulate_30_day_dataset(df_base)
