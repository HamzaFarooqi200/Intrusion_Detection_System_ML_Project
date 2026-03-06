# src/train.py

import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

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


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{name} Results")
    print("----------------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return f1


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

    print("Training models...")

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        val_f1 = f1_score(y_val, y_val_pred, average="macro")

        print(f"{name} Validation F1: {val_f1:.4f}")

        if val_f1 > best_score:
            best_score = val_f1
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name}")
    print(f"Validation F1: {best_score:.4f}")

    print("\nEvaluating best model on test set...")
    evaluate_model(best_model, X_test, y_test, best_name)

    # Feature importance (only for RandomForest)
    if best_name == "RandomForest":
        importances = best_model.feature_importances_
        print("\nTop 10 Important Features:")
        print(np.argsort(importances)[-10:])

    # Save model
    joblib.dump(best_model, "best_model.pkl")
    print("\nModel saved as best_model.pkl")


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