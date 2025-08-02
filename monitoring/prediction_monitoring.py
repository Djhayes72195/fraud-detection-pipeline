import json
import time
import math
from pathlib import Path
from statistics import mean
from scipy.stats import norm
from collections import deque
import pandas as pd
from monitoring.retraining_trigger import trigger_retraining

BUFFER_SIZE = 10_000
MIN_RECORDS_FOR_DRIFT_CHECK = 1_000
DRIFT_CHECK_INTERVAL = 100
LATENCY_THRESHOLD = .01

ALPHA = .025  # For Drift detection hyp test


def monitor_loop(log_path: Path, baseline_stats: dict, poll_interval=5):
    record_buffer = deque(maxlen=BUFFER_SIZE)
    records_since_last_check = 0
    seen_ids = set()
    print("Monitoring service is online.")

    if not log_path.exists():
        print(f"No predictions to analyze. Waiting {poll_interval} seconds...")

    while True:
        new_entries = identify_new_entries(log_path, seen_ids)  # Updates `seen_ids`

        if new_entries:
            monitor_latency(new_entries)

        for entry in new_entries:
            record_buffer.append(entry)

        records_since_last_check += len(new_entries)

        if (
            records_since_last_check > DRIFT_CHECK_INTERVAL and
            len(record_buffer) > MIN_RECORDS_FOR_DRIFT_CHECK
        ):
            records_since_last_check = 0

            summary_stats = calc_summary_stats(record_buffer)
            hyp_test_results = run_drift_hyp_test(summary_stats, baseline_stats)
            if check_if_should_retrain(hyp_test_results):
                trigger_retraining()

        time.sleep(poll_interval)


def check_if_should_retrain(hyp_test_results):
    for feature, result in hyp_test_results.items():
        if result.get("significant_drift"):
            p_value = result.get("p_value")
            print(f"SIGNIFICANT DRIFT DETECTED for feature {feature} with p-value = {p_value}")
            print("Triggering retraining")
            return True
    return False


def run_drift_hyp_test(summary_stats, baseline_stats):
    """
    Check if should retrain due to drift using Hypothesis test
    
    H_0: The mean of the feature hasn't changed
    H_1: The mean of the feature has changed (two-tailed)

    Method:
        - Set p = .05
        - Calc standard_error = observed_sd / sqrt(n)
            - Note: We can assume this is normally distributed by CLT
        - Calc z-score = (observed_mean - baseline_mean) / standard_error
        - If normal_cdf(-abs(z-score)) < .025
            - ---> send retraining trigger
    
    NOTE: We have ~20 features. This would be a good opportunity
    to implement Bonferroni correction to reduce the probability of
    Type I errors.
    """
    results = {}
    for feature, stats in summary_stats.items():

        feature_baseline_mean = (
            baseline_stats.get(feature, "Unknown").get("mean", "Unknown")
        )
        if feature_baseline_mean == "Unknown":
            print(
                f"ALERT! Missing baseline mean for {feature}. "
                "Drift detection can't be performed. Please investigate."
            )
            continue
        record_count = stats.get("record_count")
        std_error = stats.get("std_error")
        live_mean = stats.get("mean")
        z_score = (
            feature_baseline_mean - live_mean
        ) / std_error
        p_value = norm.cdf(-abs(z_score))
        drift_is_significant = p_value < ALPHA

        # NOTE: We are recording more than we really need to send a retrain trigger
        # based on the hyp test in case we want to expand reporting / observability
        # in the future.
        results[feature] = {
            "significant_drift": drift_is_significant,
            "z_score": z_score,
            "p_value": p_value,
            "baseline_mean": feature_baseline_mean,
            "live_mean": live_mean,
            "standard_error": std_error,
            "record_count": record_count
        }

    return results

def identify_new_entries(log_path: Path, seen_ids: set) -> list:
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

def monitor_latency(new_entries):

    latencies = [entry["latency_sec"] for entry in new_entries]

    high_latency_count = sum(
        1 for latency in latencies if latency > LATENCY_THRESHOLD
    )

    if high_latency_count > 0:
        avg_latency = mean(latencies)
        max_latency = max(latencies)
        print(f"LATENCY ALERT: {high_latency_count}/{len(latencies)} requests > {LATENCY_THRESHOLD}s")
        print(f"Avg: {avg_latency:.3f}s, Max: {max_latency:.3f}s")

def calc_summary_stats(record_buffer: list) -> dict:
    transactions = [record["transaction"] for record in record_buffer]
    record_count = len(transactions)
    df = pd.DataFrame(transactions)
    summary_stats = {}

    for col in df.columns:

        live_mean = df[col].mean()
        live_standard_deviation = df[col].std()
        live_standard_error = (
            live_standard_deviation /
            math.sqrt(record_count)
        )
        summary_stats[col] = {
            "mean": live_mean,
            # std_deviation isn't strictly required for downstream hyp test.
            # It is included anyway in case 
            "std_deviation": live_standard_deviation,
            "std_error": live_standard_error,
            "record_count": record_count
        }

    return summary_stats

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
