"""
Monitor logs in realtime.


"""

import json
from pathlib import Path
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

            print(f"\n--- Metrics Update ({len(new_entries)} new) ---")
            print(f"Total predictions: {len(predictions)}")
            print(f"Fraud ratio: {sum(predictions) / len(predictions):.3f}")
            print(f"Avg latency: {mean(latencies):.3f} sec")
            print(f"Model versions: {versions}")

        time.sleep(poll_interval)

if __name__ == "__main__":
    monitor_log(Path("C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\fraud-detection-pipeline\\logs\\stream_day5.jsonl"))
