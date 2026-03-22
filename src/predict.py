"""Predict: Model prediction module."""

import os
import sys

import joblib
import pandas as pd

# Ensure local imports work when called from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import load_and_preprocess_data

# Load trained model
model = joblib.load("models/xgb_model.pkl")


def predict_sample(sample_df):
    """Predict readmission probability for a given sample DataFrame."""
    if not isinstance(sample_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    prob = model.predict_proba(sample_df)[:, 1]
    return prob[0]


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    sample = X.iloc[[0]]
    prediction = predict_sample(sample)
    print(f"Predicted readmission risk: {prediction:.4f}")