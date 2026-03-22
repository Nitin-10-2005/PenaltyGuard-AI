"""Finance: Financial impact and penalty calculation."""

import joblib


def calculate_penalty(predicted_risk, hospital_revenue):
    """Calculate excess readmission ratio (ERR) and estimated penalty."""

    baseline = joblib.load("models/baseline.pkl")
    if baseline > 0.3:
        expected_baseline_risk = 0.2
    else:
        expected_baseline_risk = baseline

    # Guard against zero baseline
    if expected_baseline_risk <= 0:
        expected_baseline_risk = 0.15

    # Excess Readmission Ratio
    err = predicted_risk / expected_baseline_risk

    # Penalty calculation
    if predicted_risk < expected_baseline_risk:
        penalty = 0
        return err, penalty

    # Add buffer zone
    if predicted_risk <= expected_baseline_risk + 0.05:
        penalty = 0
        return err, penalty

    # Proportional penalty logic
    # Penalty rate scales linearly: 0% at ERR=1.0 → 3% cap at ERR≥1.5
    max_penalty_rate = 0.03
    err_cap = 1.5  # ERR at which penalty maxes out

    if err <= 1:
        penalty = 0
    else:
        excess = err - 1.0
        penalty_rate = min(excess / (err_cap - 1.0), 1.0) * max_penalty_rate
        penalty = penalty_rate * hospital_revenue

    return err, penalty


if __name__ == "__main__":
    hospital_revenue = 10000000  # 1 crore

    # Test across risk levels
    for sample_risk in [0.15, 0.25, 0.35, 0.50, 0.80]:
        err, penalty = calculate_penalty(sample_risk, hospital_revenue)
        print(f"Risk: {sample_risk:.2f}  →  ERR: {err:.2f}  →  Penalty: ₹{penalty:,.0f}")