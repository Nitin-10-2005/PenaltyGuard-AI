"""Train: XGBoost model training pipeline with hyperparameter tuning."""

import os
import sys

import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Ensure local imports work when called from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import load_and_preprocess_data


def train_model():
    """Load data, tune XGBoost via GridSearchCV, evaluate, and save best model."""

    # Load preprocessed data
    X, y = load_and_preprocess_data()

    # Compute and save baseline risk
    baseline_risk = y.mean()
    print(f"Baseline Risk: {baseline_risk:.4f}")
    joblib.dump(baseline_risk, "models/baseline.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

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
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Best Parameters: {grid.best_params_}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Save best model
    joblib.dump(best_model, "models/xgb_model.pkl")
    print("Best model saved to models/xgb_model.pkl")


if __name__ == "__main__":
    train_model()
