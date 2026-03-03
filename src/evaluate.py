# src/evaluate.py

import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from train import load_data


def main():
    print("Loading model...")
    model = joblib.load("best_model.pkl")

    print("Loading data...")
    X, y = load_data("data/kddcup.data_10_percent.gz")

    y_pred = model.predict(X)

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    main()