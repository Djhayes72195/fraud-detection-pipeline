from pathlib import Path

LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "model_registry"

TRAIN_AND_VAL_IDS_PATH = Path(__file__).resolve().parents[1] / "metadata" / "train_sids.json"