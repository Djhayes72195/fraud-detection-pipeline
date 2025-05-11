from fastapi import FastAPI
from .schemas import Transaction
from inference.predict import predict_and_log
from modeling.model_io import load_by_day

model = load_by_day(day=5)  # load once at startup
app = FastAPI()

@app.post("/predict")
def predict(transaction: Transaction):
    pred = predict_and_log(transaction.dict(), model, threshold=0.8)
    return {"prediction": pred}
