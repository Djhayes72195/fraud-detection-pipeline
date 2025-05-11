from .config import LOCAL_MODEL_PATH
from pathlib import Path
import xgboost as xgb

def save(model, day: int, destination="local"):
    if destination != "local":
        raise NotImplementedError(f"Saving to '{destination}' is not supported yet.")
    
    path = Path(LOCAL_MODEL_PATH) / f"model_day_{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model.save_model(str(path))
    print(f"Model saved to {path.resolve()}")

def load_by_day(day: int, source="local"):
    if source != "local":
        raise NotImplementedError(f"Saving to '{source}' is not supported yet.")
    path = Path(LOCAL_MODEL_PATH) / f"model_day_{day}.json"
    if not path.exists():
        raise IOError("Attempting to access a model that does not exist")

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


        
