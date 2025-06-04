from fastapi import FastAPI
from .schemas import Transaction
from pathlib import Path
import json
from inference.predict import predict_and_log
from modeling.model_io import load_by_day

# uvicorn api.app:app --reload

LOG_PATH = Path("logs/stream_day5.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, "a")  # append mode

def api_logger(log_entry):
    log_file.write(json.dumps(log_entry) + "\n")
    log_file.flush()  # Forces to empty buffer and write to disk, avoids lost logs


model = load_by_day(day=5)  # load once at startup
app = FastAPI()

@app.post("/predict")
def predict(transaction: Transaction):
    pred = predict_and_log(transaction.dict(), model, threshold=0.8, logger=api_logger)
    return {"prediction": pred}
