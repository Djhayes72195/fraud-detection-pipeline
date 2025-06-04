# inference/predict.py

import pandas as pd
import datetime
import time
from datetime import datetime

def predict_and_log(transaction: dict, model, threshold=0.8, logger=None):
    df = pd.DataFrame([transaction])

    start_time = time.time()

    prob = model.predict_proba(df)[0, 1]

    latency_sec = time.time() - start_time

    pred = int(prob > threshold)

    log_entry = {
        "transaction": transaction,
        "prediction": pred,
        "threshold": threshold,
        "latency_sec": latency_sec,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Log the result
    if logger:
        logger(log_entry)
    else:
        print(log_entry)

    return pred
