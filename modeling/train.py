import pandas as pd
import os
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .config import TRAIN_AND_VAL_IDS_PATH

def train(df):
    X = df.drop(columns=["Class", "Day", "PerturbationScheme", "dt"])
    y = df[["Class", "sid"]]

    X_train, X_val, X_test, y_train, y_val, y_test = _persistent_data_split(X, y)

    # Allows XGBoost to give higher weight to fraud cases
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    # TODO: Switch on when out of prototyping phase
    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]

    # y_pred = model.predict(X_val)
    threshold = .5
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print(classification_report(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_prob))

    return model


def _persistent_data_split(X, y):
    if os.path.exists(TRAIN_AND_VAL_IDS_PATH):
        with open(TRAIN_AND_VAL_IDS_PATH, 'r') as f:
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

        with open(TRAIN_AND_VAL_IDS_PATH, "w") as f:
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
