from .config import LOCAL_MODEL_PATH
from pathlib import Path
from modeling.versioned_model import VersionedModel
import xgboost as xgb

def save(model, day: int, destination="local"):
    if destination != "local":
        raise NotImplementedError(f"Saving to '{destination}' is not supported yet.")
    
    path = Path(LOCAL_MODEL_PATH) / f"day={day}" / f"{model.version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model.save_model(str(path))
    print(f"Model saved to {path.resolve()}")

def load_by_day(day: int, source="local"):
    if source != "local":
        raise NotImplementedError(f"Saving to '{source}' is not supported yet.")
    path_to_day_models = Path(LOCAL_MODEL_PATH) / f"day={day}"

    path, model_id = find_first_model_in_day_dir(path_to_day_models)
    if not path.exists():
        raise IOError("Attempting to access a model that does not exist")

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return VersionedModel(model, model_id)


def find_first_model_in_day_dir(metadata_dir):
    metadata_dir = Path(metadata_dir)
    json_files = list(metadata_dir.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {metadata_dir}")
    
    # Pick the first one (sorted by name, just to make it deterministic)
    json_file = sorted(json_files)[0]
    model_id_found = json_file.stem
    return json_file, model_id_found


        
