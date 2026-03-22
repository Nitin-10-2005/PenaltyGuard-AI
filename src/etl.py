"""ETL: Extract, Transform, Load pipeline for hospital readmission data."""

import os
import joblib
import pandas as pd


CATEGORICAL_COLS = [
    "season",
    "gender",
    "region",
    "primary_diagnosis",
    "treatment_type",
    "insurance_type",
    "discharge_disposition",
]


def load_data():
    """Load dataset, drop IDs, separate features and target."""
    df = pd.read_csv("data/hospital_readmission_dataset.csv")
    df = df.drop(columns=["patient_id", "admission_date"])

    y = df["label"]
    X = df.drop(columns=["label"])

    return X, y


def fit_transform(X):
    """Compute imputation stats on X, save them, and return preprocessed X.

    Use this on TRAINING data only. Saves medians, modes, and column list
    to models/ so they can be reused at inference time.
    """
    numerical_cols = [col for col in X.columns if col not in CATEGORICAL_COLS]

    # Compute stats on this data (should be training set)
    medians = {col: X[col].median() for col in numerical_cols}
    modes = {col: X[col].mode()[0] for col in CATEGORICAL_COLS}

    # Save stats for reuse at inference time
    joblib.dump({"medians": medians, "modes": modes}, "models/preprocess_stats.pkl")

    # Apply imputation
    X = _apply_imputation(X, medians, modes, numerical_cols)

    # One-hot encode
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    X = X.astype(float)

    # Save column order for alignment
    joblib.dump(list(X.columns), "models/columns.pkl")

    assert X.isnull().sum().sum() == 0, "Missing values remain in features"
    return X


def transform(X):
    """Apply saved preprocessing stats to new data (test set or live input).

    Loads medians, modes, and column list from models/ that were saved
    during fit_transform().
    """
    stats = joblib.load("models/preprocess_stats.pkl")
    columns = joblib.load("models/columns.pkl")

    medians = stats["medians"]
    modes = stats["modes"]
    numerical_cols = [col for col in X.columns if col not in CATEGORICAL_COLS]

    # Apply imputation using TRAINING stats
    X = _apply_imputation(X, medians, modes, numerical_cols)

    # One-hot encode and align columns to training set
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    X = X.reindex(columns=columns, fill_value=0)
    X = X.astype(float)

    return X


def _apply_imputation(X, medians, modes, numerical_cols):
    """Fill missing values using provided stats."""
    X = X.copy()
    for col in numerical_cols:
        if col in medians:
            X[col] = X[col].fillna(medians[col])
    for col in CATEGORICAL_COLS:
        if col in modes:
            X[col] = X[col].fillna(modes[col])
    return X


def load_and_preprocess_data():
    """Convenience: load + preprocess entire dataset (for quick testing).

    NOTE: This fits imputation on ALL data. For proper train/test separation,
    use load_data() + fit_transform() / transform() instead.
    """
    X, y = load_data()
    X = fit_transform(X)
    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
