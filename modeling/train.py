import os
import hashlib
import datetime
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .config import TRAIN_AND_VAL_SIDS_PATH, TRAIN_IDS_PATH, MODEL_METADATA_PATH

import os
import hashlib
from datetime import datetime
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .config import TRAIN_AND_VAL_SIDS_PATH, TRAIN_IDS_PATH, MODEL_METADATA_PATH

def make_model_id(hyperparams, train_ids_hash):
    """Create a unique model ID using hyperparams and train data hash"""
    to_hash = {
        "hyperparameters": hyperparams,
        "train_ids_hash": train_ids_hash
    }
    json_str = json.dumps(to_hash, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()

def train(df):
    transaction_ids = df[["TransactionID", "sid"]]

    X = df.drop(columns=["Class", "Day", "PerturbationScheme", "dt", "TransactionID"])
    y = df[["Class", "sid"]]

    X_train, X_val, X_test, y_train, y_val, y_test = _persistent_data_split(X, y)

    train_ids = list(transaction_ids.loc[X_train.index, "TransactionID"])

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    hyperparams = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42
    }

    model = xgb.XGBClassifier(**hyperparams)

    model.fit(X_train, y_train)

    # Validation
    threshold = .5
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print(classification_report(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_prob))

    model_id = write_metadata(train_ids, hyperparams)

    model.version = model_id

    return model

def write_metadata(train_ids, hyperparams):
    """
    Write metadata related to the training process.

    This function writes:
        - A JSON file containing the training IDs.
        - A JSON file containing the model metadata.

    Both files are identified using generated hashes.

    The function returns the model ID, which serves as the version identifier for this model.
    Each prediction will be logged with the model version, enabling traceability back to
    the exact data and configuration used during training.
    """
    train_ids_hash = hashlib.md5(json.dumps(train_ids, sort_keys=True).encode()).hexdigest()

    train_ids_file_path = TRAIN_IDS_PATH / f"train_ids_{train_ids_hash}.json"
    with open(train_ids_file_path, "w") as f:
        json.dump(train_ids, f, indent=4)

    model_id = make_model_id(hyperparams, train_ids_hash)

    model_metadata = {
        "model_id": model_id,
        "hyperparameters": hyperparams,
        "train_ids_hash": train_ids_hash,
        "train_timestamp": datetime.utcnow().isoformat()
    }

    model_metadata_file_path = MODEL_METADATA_PATH / f"model_{model_id}.json"
    with open(model_metadata_file_path, "w") as f:
        json.dump(model_metadata, f, indent=4)
    return model_id



def _persistent_data_split(X, y):
    if os.path.exists(TRAIN_AND_VAL_SIDS_PATH):
        with open(TRAIN_AND_VAL_SIDS_PATH, 'r') as f:
            saved_ids = json.load(f)
        train_sids = set(saved_ids["train"])
        val_sids = set(saved_ids["val"])

        X_train = X[X["sid"].isin(train_sids)]
        X_val = X[X["sid"].isin(val_sids)]
        X_test = X[~X["sid"].isin(train_sids.union(val_sids))]

        y_train = y[y["sid"].isin(train_sids)]
        y_val = y[y["sid"].isin(val_sids)]
        y_test = y[~y["sid"].isin(train_sids.union(val_sids))]

    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y["Class"], random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp["Class"], random_state=42
        )

        train_sids = list(set(y_train["sid"]))
        val_sids = list(set(y_val["sid"]))

        with open(TRAIN_AND_VAL_SIDS_PATH, "w") as f:
            json.dump({"train": train_sids, "val": val_sids}, f, indent=4)

    for df in [X_train, X_val, X_test, y_train, y_val, y_test]:
        df.drop(columns=["sid"], inplace=True)

    return (
        X_train,
        X_val,
        X_test,
        y_train["Class"],
        y_val["Class"],
        y_test["Class"]
    )
