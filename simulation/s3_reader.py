import time
import json
import requests
from data_processing.data_ingestion import load_days
from .constants import GROUND_TRUTH_LOGPATH
import threading

write_lock = threading.Lock()


def stream_transactions(day: int, api_url: str, delay: float = 0.1):
    df = load_days(day, day)

    with requests.Session() as session:
        for _, row in df.iterrows():
            start = time.time()
            write_ground_truth(row)
            transaction = row.drop(["Class", "sid"]).to_dict()
            print(f"Transaction {transaction['TransactionID']} loaded from source.")
            try:
                response = session.post(api_url, json=transaction)  # This line takes too long
                print(f"→ {response.status_code}: {response.json() if response.ok else response.text}")
            except Exception as e:
                print(f"Request failed: {e}")

            time.sleep(delay)
            end = time.time()
            print(f"Stream transaction and predicty latency: {end - start}")

def write_ground_truth(row, delay_sec: int = 15):
    entry = {
        "TransactionID": row["TransactionID"],
        "sid": row["sid"],
        "Class": row["Class"],
        "label_available_at": time.time() + delay_sec
    }
    line = json.dumps(entry) + "\n"
    with write_lock, open(GROUND_TRUTH_LOGPATH, "a") as f:
        f.write(line)
        f.flush()
