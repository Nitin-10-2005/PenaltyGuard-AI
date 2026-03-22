"""Train: XGBoost model training pipeline with hyperparameter tuning."""

import os
import sys

import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# Ensure local imports work when called from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import load_data, fit_transform, transform


def train_model():
    """Load data, tune XGBoost via GridSearchCV, evaluate, and save best model."""

    # Load raw data (no preprocessing yet)
    X, y = load_data()

    # Compute and save baseline risk
    baseline_risk = y.mean()
    print(f"Baseline Risk: {baseline_risk:.4f}")
    joblib.dump(baseline_risk, "models/baseline.pkl")

    # Split BEFORE preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit preprocessing on training data only, then transform test data
    X_train = fit_transform(X_train)
    X_test = transform(X_test)

    # Handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # Base model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    # Parameter grid
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # Grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_

    # Evaluate
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nBest Parameters: {grid.best_params_}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save best model
    joblib.dump(best_model, "models/xgb_model.pkl")
    print("\nBest model saved to models/xgb_model.pkl")


if __name__ == "__main__":
    train_model()