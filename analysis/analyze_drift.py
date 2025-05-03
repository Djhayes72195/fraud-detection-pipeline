import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "raw_data"
REPORT_DIR = Path(__file__).resolve().parents[1] / "analysis" / "drift_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = [5, 15, 25]
FEATURES = ["Amount", "Time", "V4", "V11"]

def load_day(day):
    path = ROOT / f"2021-01-{day+1:02d}" / "transactions.parquet"
    return pd.read_parquet(path)

def plot_feature_distributions():
    for feature in FEATURES:
        plt.figure(figsize=(8, 4))
        for day in DAYS:
            df = load_day(day)
            # Optional: exclude outliers
            df = df[df[feature] < df[feature].quantile(0.99)]
            plt.hist(df[feature], bins=100, alpha=0.4, label=f"Day {day}", density=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        out_path = REPORT_DIR / f"{feature}_distribution.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot: {out_path}")

def save_class_ratios():
    ratios = []
    for day in DAYS:
        df = load_day(day)
        ratio = df["Class"].mean()
        ratios.append({"Day": day, "FraudRate": ratio, "RowCount": len(df)})
    df_ratios = pd.DataFrame(ratios)
    out_path = REPORT_DIR / "fraud_ratios.csv"
    df_ratios.to_csv(out_path, index=False)
    print(f"Saved fraud class ratios to: {out_path}")

if __name__ == "__main__":
    plot_feature_distributions()
    save_class_ratios()
