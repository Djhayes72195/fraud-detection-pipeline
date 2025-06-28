"""
Monitor logs in realtime.


"""

import json
from pathlib import Path
import pandas as pd
from statistics import mean
import time

def monitor_log(log_path: Path, poll_interval=5):
    seen = set()

    while True:
        with open(log_path, "r") as f:
            lines = f.readlines()

        new_entries = [line for line in lines if line not in seen]
        seen.update(new_entries)

        if new_entries:
            predictions = []
            latencies = []
            transactions = []
            versions = set()

            for line in new_entries:
                entry = json.loads(line)
                transactions.append(entry['transaction'])
                predictions.append(entry["prediction"])
                latencies.append(entry["latency_sec"])
                versions.add(entry.get("model_version", "unknown"))

            drift_scores = calculate_drift(transactions)

            print(f"\n--- Metrics Update ({len(new_entries)} new) ---")
            print(f"Total predictions: {len(predictions)}")
            print(f"Fraud ratio: {sum(predictions) / len(predictions):.3f}")
            print(f"Avg latency: {mean(latencies):.3f} sec")
            print(f"Model versions: {versions}")

        time.sleep(poll_interval)


def calculate_drift(transactions):
    """Calculate data drift"""
    df = pd.DataFrame(transactions)
    drift_scores = {}

    for col in df.columns:
        if col not in baseline_stats:
            continue
        live_mean = df[col].mean()
        base_mean = baseline_stats[col]["mean"]
        base_std = baseline_stats[col]["std"]

        if base_std > 0:
            drift_score = abs(live_mean - base_mean) / base_std
            drift_scores[col] = drift_score

    return drift_scores


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1] / "logs"
    baseline_stats_path = base_path / "baseline_stats.json"
    streaming_logs_path = base_path / "stream_day5.jsonl"

    with open(baseline_stats_path, "r") as f:
        baseline_stats = json.load(f)

    streaming_logs_path = base_path / "stream_day5.jsonl"
    monitor_log(streaming_logs_path)
