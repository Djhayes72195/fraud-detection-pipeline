import json
import time
from pathlib import Path
from statistics import mean
import pandas as pd

def monitor_loop(log_path: Path, baseline_stats: dict, poll_interval=5):
    seen_ids = set()
    print("Monitoring service is online.")

    if not log_path.exists():
        print(f"No predictions to analyze. Waiting {poll_interval} seconds...")

    while True:
        new_entries = identify_new_entries(log_path, seen_ids)  # Updates `seen_ids`

        if new_entries:
            predictions, latencies, transactions, versions = parse_new_entries(new_entries)
            drift_scores = calculate_drift(transactions, baseline_stats)
            analyze_drift(drift_scores)
            report_metrics(new_entries, predictions, latencies, versions)

        time.sleep(poll_interval)

def identify_new_entries(log_path: Path, seen_ids: set):
    new_entries = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                txn_id = entry.get("transaction_id")
                if txn_id and txn_id not in seen_ids:
                    seen_ids.add(txn_id)
                    new_entries.append(entry)
            except json.JSONDecodeError:
                continue
    return new_entries

def parse_new_entries(new_entries):
    predictions = []
    latencies = []
    transactions = []
    versions = set()

    for entry in new_entries:
        transactions.append(entry["transaction"])
        predictions.append(entry["prediction"])
        latencies.append(entry["latency_sec"])
        versions.add(entry.get("model_version", "unknown"))

    return predictions, latencies, transactions, versions

def calculate_drift(transactions: list, baseline_stats: dict):
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

def analyze_drift(drift_scores):
    print("Reached analyze_drift func")

def report_metrics(new_entries, predictions, latencies, versions):
    print(f"\n--- Metrics Update ({len(new_entries)} new) ---")
    print(f"Total predictions: {len(predictions)}")
    print(f"Fraud ratio: {sum(predictions) / len(predictions):.3f}")
    print(f"Avg latency: {mean(latencies):.3f} sec")
    print(f"Model versions: {versions}")

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1] / "logs"
    baseline_stats_path = base_path / "baseline_stats.json"
    log_path = base_path / "stream_day5.jsonl"

    with open(baseline_stats_path, "r") as f:
        baseline_stats = json.load(f)

    monitor_loop(log_path, baseline_stats)
