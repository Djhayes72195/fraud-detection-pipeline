from fastapi import FastAPI
from retrainer.controller import retrain_pipeline


app = FastAPI()

@app.post("/retrain")
def trigger_retraining():
    try:
        retrain_pipeline()
        return {"status": "success", "message": "Retraining completed."}
    except Exception as e:
        return {"status": "error", "message": str(e)}