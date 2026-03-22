"""Finance: Financial impact and penalty calculation."""

import joblib


def calculate_penalty(predicted_risk, hospital_revenue):
    """Calculate excess readmission ratio (ERR) and estimated penalty."""

    baseline = joblib.load("models/baseline.pkl")
    if baseline > 0.3:
        expected_baseline_risk = 0.2
    else:
        expected_baseline_risk = baseline
    # Excess Readmission Ratio
    err = predicted_risk / expected_baseline_risk

    # Penalty calculation
    if err <= 1:
        penalty = 0
    else:
        penalty_rate = max(0, err - 1)
        # scale it realistically but keep cap
        penalty_rate = min(penalty_rate * 0.1, 0.03)

        penalty = penalty_rate * hospital_revenue

    return err, penalty


if __name__ == "__main__":
    sample_risk = 0.25
    hospital_revenue = 10000000  # 1 crore

    err, penalty = calculate_penalty(sample_risk, hospital_revenue)

    print(f"ERR: {err:.2f}")
    print(f"Penalty: ₹{penalty:,.2f}")
