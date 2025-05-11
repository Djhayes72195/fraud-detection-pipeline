# inference/predict.py

import pandas as pd
import datetime

def predict_and_log(transaction: dict, model, threshold=0.5, logger=None):
    df = pd.DataFrame([transaction])

    prob = model.predict_proba(df)[0, 1]
    pred = int(prob > threshold)

    # Construct log entry
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": transaction,
        "probability": prob,
        "prediction": pred,
        "threshold": threshold
    }

    # Log the result
    if logger:
        logger(log_entry)
    else:
        print(log_entry)

    return pred
