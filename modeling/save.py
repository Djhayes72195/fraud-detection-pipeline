from .config import LOCAL_MODEL_PATH
from pathlib import Path

def save(model, day: int, destination="local"):
    if destination != "local":
        raise NotImplementedError(f"Saving to '{destination}' is not supported yet.")
    
    path = Path(LOCAL_MODEL_PATH) / f"model_day_{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model.save_model(str(path))
    print(f"Model saved to {path.resolve()}")


        
