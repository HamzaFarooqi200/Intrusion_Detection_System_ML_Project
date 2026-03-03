# src/train.py

import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utils import set_seed


def load_data(path):
    df = pd.read_csv(path, header=None)

    # Last column is label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode categorical columns
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y


def main(args):
    set_seed(42)

    print("Loading data...")
    X, y = load_data(args.data_path)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Running GridSearch...")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_val_pred = best_model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    print(f"Validation F1: {val_f1:.4f}")
    print(f"Best Params: {grid.best_params_}")

    # Save model
    joblib.dump(best_model, "best_model.pkl")
    print("Model saved as best_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/kddcup.data_10_percent.gz",
        help="Path to dataset"
    )
    args = parser.parse_args()
    main(args)