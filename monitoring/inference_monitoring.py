import json
import time
from pathlib import Path
from collections import deque
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
from modeling.config import PERFORMANCE_METRICS_PATH
from monitoring.constants import (
    MIN_PRED_BEFORE_INFERENCE_MONITORING,
    RETRAIN_THRESHOLDS,
    MIN_FRAUD_CASES_BEFORE_INFERENCE_MONITORING
)
from monitoring.retraining_trigger import trigger_retraining

def monitor_loop(ground_truths_path: Path, predicted_transactions_path: Path, time_interval: int = 60):
    # patched_df = pd.DataFrame()
    record_buffer = deque(maxlen=100_000)
    patched_records = set()
    print("Monitoring service is online.")

    while True:
        predictions: pd.DataFrame = parse_transaction_data(
            load_jsonl(predicted_transactions_path)
        )
        ground_truths = pd.DataFrame(
            load_jsonl(ground_truths_path)
        )
        if predictions.empty or ground_truths.empty:
            time.sleep(time_interval)

        new_predictions = predictions[
            ~predictions["TransactionID"].isin(patched_records)
        ].drop_duplicates("TransactionID")
        new_ground_truths = ground_truths[
            ~ground_truths["TransactionID"].isin(patched_records)
        ].drop_duplicates("TransactionID")

        new_patched_df = new_ground_truths.merge(new_predictions, on="TransactionID", how="inner")
        patched_records.update(new_patched_df["TransactionID"])
        # fraud_cases_seen += new_patched_df["Class"].sum()

        for row in new_patched_df.to_dict("records"):
            record_buffer.append(row)

        buffer_df = pd.DataFrame(record_buffer)
        if buffer_df.empty:
            print("Inference monitor buffer is empty. Skipping.")
            time.sleep(time_interval)
            continue


        buffer_df, most_recent_model_version = filter_to_most_recent_model(buffer_df)

        fraud_cases_seen = buffer_df["Class"].sum()
        if not thresholds_met(buffer_df, fraud_cases_seen):
            time.sleep(time_interval)
            continue

        base_performance_metrics = look_up_base_perf_metrics(most_recent_model_version)
        current_metrics = calculate_current_metrics(buffer_df)

        should_retrain = check_if_should_retrain(base_performance_metrics, current_metrics)
        if should_retrain:
            trigger_retraining()

        time.sleep(time_interval)

def thresholds_met(df, fraud_cases_seen):
    if len(df) < MIN_PRED_BEFORE_INFERENCE_MONITORING:
        print(f"Inference monitoring was called with <{MIN_PRED_BEFORE_INFERENCE_MONITORING} records. Skipping evaluation.")
        return False
    if fraud_cases_seen < MIN_FRAUD_CASES_BEFORE_INFERENCE_MONITORING:
        print(f"Inference monitoring was called with <{MIN_FRAUD_CASES_BEFORE_INFERENCE_MONITORING} true fraud cases. Skipping evaluation.")
        return False
    return True


def check_if_should_retrain(base_metrics, current_metrics):
    for metric, threshold in RETRAIN_THRESHOLDS.items():
        if base_metrics[metric] - current_metrics[metric] > threshold:
            print("Performance has degraded ---> sending retraining trigger.")
            return True
    print("Performance has not degraded significantly. Continuing without retraining.")
    return False


def calculate_current_metrics(patched_df):
    y_prob = patched_df["probability"]
    y_pred = patched_df["prediction"]
    truth = patched_df["Class"]
    return {
        "accuracy": (y_pred == truth).mean(),
        "precision": precision_score(truth, y_pred),
        "recall": recall_score(truth, y_pred),
        "f1": f1_score(truth, y_pred),
        "roc_auc": roc_auc_score(truth, y_prob)
    }


def filter_to_most_recent_model(patched_df):
    """
    Filter records down to the most recent model used.

    The purpose of this service is to schedule retraining
    if performance decays. Only records originating from
    the current production model are relevant.
    """
    most_recent_model = patched_df.sort_values(
        by="prediction_timestamp"
    ).iloc[-1]["model_version"]
    patched_df = patched_df[patched_df["model_version"] == most_recent_model]
    return patched_df, most_recent_model


def load_jsonl(path: Path):
    if not path.exists():
        print(f"{path} does not exist. Returning empty list.")
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_transaction_data(transactions: dict) -> pd.DataFrame:
    parsed_transactions = []
    for transaction in transactions:
        flat_record = transaction['transaction'].copy()
        flat_record['TransactionID'] = transaction['transaction_id']
        flat_record['prediction'] = transaction["prediction"]
        flat_record["probability"] = transaction["probability"]
        flat_record["model_version"] = transaction["model_version"]
        flat_record["prediction_timestamp"] = transaction["timestamp"]
        parsed_transactions.append(flat_record)
    return pd.DataFrame(parsed_transactions)


def look_up_base_perf_metrics(most_recent_model_version):
    base_metrics_path = PERFORMANCE_METRICS_PATH / f"metrics_{most_recent_model_version}.json"
    if not base_metrics_path.exists():
        print(f"Baseline metrics can not be recovered for model: {base_metrics_path}. Please investigate.")
        raise FileNotFoundError
    with open(base_metrics_path, "r") as f:
        base_metrics = json.load(f)
    return base_metrics

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1] / "logs"
    ground_truths_path = base_path / "ground_truths_released.jsonl"
    predicted_transactions_path = base_path / "stream_day5.jsonl"

    monitor_loop(ground_truths_path, predicted_transactions_path)
