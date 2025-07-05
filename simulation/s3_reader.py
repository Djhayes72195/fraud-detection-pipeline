import time
import json
import requests
from data_processing.data_ingestion import load_days
from .constants import GROUND_TRUTH_LOGPATH

def stream_transactions(day: int, api_url: str, delay: float = 0.1):
    df = load_days(day, day)

    for _, row in df.iterrows():
        write_ground_truth(row)
        transaction = row.drop(["Class", "sid"]).to_dict()
        print(f"Transaction {transaction['TransactionID']} loaded from source.")
        try:
            response = requests.post(api_url, json=transaction)
            print(f"→ {response.status_code}: {response.json() if response.ok else response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(delay)

def write_ground_truth(row, delay_sec: int = 15):
    """
    Write ground truth labels.

    Ground truth labels will be released after a delay.
    """
    entry = {
        "TransactionID": row["TransactionID"],
        "sid": row["sid"],
        "Class": row["Class"],
        "label_available_at": time.time() + delay_sec
    }
    with open(GROUND_TRUTH_LOGPATH, "a") as f:
        f.write(json.dumps(entry) + "\n")