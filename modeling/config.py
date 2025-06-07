from pathlib import Path

LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "model_registry"

METADATA_PATH = Path(__file__).resolve().parents[1] / "metadata"

# Data Simulation tracker IDs. Simulated data which shares the same original data parent
# have the same sid. This resource is persistent.
TRAIN_AND_VAL_SIDS_PATH = METADATA_PATH / "train_sids.json"

# Used for storing unique training IDs for audit purposes.
TRAIN_IDS_PATH = METADATA_PATH / "train_ids"

# Used for storing hyperparameter / metadata each time a model is trained.
MODEL_METADATA_PATH = METADATA_PATH / "models"

