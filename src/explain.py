"""Explain: Model explainability using SHAP."""

import os
import sys

import joblib
import pandas as pd
import shap

# Ensure local imports work when called from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import load_and_preprocess_data

# Load trained model and initialize explainer
model = joblib.load("models/xgb_model.pkl")
explainer = shap.TreeExplainer(model)


def explain_sample(sample_df):
    """Return top 10 SHAP feature contributions for a single sample."""
    shap_values = explainer.shap_values(sample_df)

    shap_df = pd.DataFrame({
        "feature": sample_df.columns,
        "shap_value": shap_values[0],
    })

    shap_df["abs_val"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values(by="abs_val", ascending=False)

    return shap_df.head(10)


def global_importance(X):
    """Return top 10 globally important features by mean |SHAP|."""
    shap_values = explainer.shap_values(X)
    mean_abs = abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_abs,
    }).sort_values(by="importance", ascending=False)

    return importance_df.head(10)


if __name__ == "__main__":
    X, y = load_and_preprocess_data()

    sample = X.iloc[[0]]

    print("\nLocal Explanation:")
    print(explain_sample(sample))

    print("\nGlobal Importance:")
    print(global_importance(X))
