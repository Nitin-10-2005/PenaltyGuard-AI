"""ETL: Extract, Transform, Load pipeline for hospital readmission data."""

import pandas as pd


def load_and_preprocess_data():
    """Load dataset, clean, encode, and return features and target."""

    # Load dataset
    df = pd.read_csv("data/hospital_readmission_dataset.csv")

    # Drop unnecessary columns
    df = df.drop(columns=["patient_id", "admission_date"])

    # Separate target
    y = df["label"]
    X = df.drop(columns=["label"])

    # Define column types
    categorical_cols = [
        "season",
        "gender",
        "region",
        "primary_diagnosis",
        "treatment_type",
        "insurance_type",
        "discharge_disposition",
    ]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Fill missing values — numerical with median
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())

    # Fill missing values — categorical with mode
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Ensure all columns are numeric
    X = X.astype(float)

    # Final sanity checks
    assert X.isnull().sum().sum() == 0, "Missing values remain in features"
    assert y.isnull().sum() == 0, "Missing values remain in target"

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
