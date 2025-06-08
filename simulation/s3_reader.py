import time
import requests
from data_processing.data_ingestion import load_days

def stream_transactions(day: int, api_url: str, delay: float = 0.1):
    df = load_days(day, day)

    for _, row in df.iterrows():
        transaction = row.drop(["Class", "sid"]).to_dict()
        print(f"Transaction {transaction["TransactionID"]} loaded from source.")
        try:
            response = requests.post(api_url, json=transaction)
            print(f"→ {response.status_code}: {response.json() if response.ok else response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(delay)

