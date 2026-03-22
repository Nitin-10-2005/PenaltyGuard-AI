import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import streamlit as st

from src.etl import transform
from src.predict import predict_sample
from src.explain import explain_sample
from src.finance import calculate_penalty
from src.report import generate_report

st.title("PenaltyGuard AI - Readmission Risk & Penalty System")

st.subheader("Enter Patient Details")

age = st.number_input("Age", 0, 120, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
season = st.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])
region = st.selectbox("Region", ["Central", "East", "North", "South", "West"])

comorbidities_count = st.number_input("Comorbidities Count", 0, 10, 1)
length_of_stay = st.number_input("Length of Stay", 1, 30, 5)
medications_count = st.number_input("Medications Count", 0, 20, 5)
followup_visits_last_year = st.number_input("Followups Last Year", 0, 20, 2)
prev_readmissions = st.number_input("Previous Readmissions", 0, 10, 0)

primary_diagnosis = st.selectbox("Primary Diagnosis", [
    "Appendicitis", "COPD", "Diabetes", "Fracture", "Heart Failure",
    "Hypertension", "Influenza", "Kidney Disease", "Pneumonia",
    "Sepsis", "Stroke"
])

treatment_type = st.selectbox("Treatment Type", [
    "Conservative", "Interventional", "Medical", "Surgical"
])

insurance_type = st.selectbox("Insurance Type", [
    "Medicaid", "Medicare", "Private", "Uninsured"
])

discharge_disposition = st.selectbox("Discharge Disposition", [
    "Home", "Home Health", "Rehab", "Skilled Nursing"
])

revenue = st.number_input("Hospital Revenue (₹)", value=10000000)

if st.button("Calculate Risk"):
    input_dict = {
        "age": age,
        "gender": gender,
        "season": season,
        "region": region,
        "comorbidities_count": comorbidities_count,
        "length_of_stay": length_of_stay,
        "medications_count": medications_count,
        "followup_visits_last_year": followup_visits_last_year,
        "prev_readmissions": prev_readmissions,
        "primary_diagnosis": primary_diagnosis,
        "treatment_type": treatment_type,
        "insurance_type": insurance_type,
        "discharge_disposition": discharge_disposition
    }

    input_df = pd.DataFrame([input_dict])

    # Use saved preprocessing stats and column list from training
    input_encoded = transform(input_df)

    # Prediction
    risk = predict_sample(input_encoded)

    if risk < 0.2:
        risk_level = "Low"
        color = "green"
    elif risk < 0.5:
        risk_level = "Medium"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"

    # Financial
    err, penalty = calculate_penalty(risk, revenue)

    # SHAP
    shap_df = explain_sample(input_encoded)

    st.subheader("Prediction")
    st.markdown(f"""
    ### Risk Score: {risk:.2f}
    ### Risk Level: <span style='color:{color}'>{risk_level}</span>
    """, unsafe_allow_html=True)

    st.subheader("Financial Impact")
    st.write(f"ERR: {err:.2f}")
    st.write(f"Estimated Penalty: ₹{penalty:,.2f}")

    st.subheader("Top Risk Factors (SHAP)")

    # Sort for better visualization
    shap_plot = shap_df.sort_values(by="shap_value")

    st.bar_chart(
        shap_plot.set_index("feature")["shap_value"]
    )

    st.dataframe(shap_df)

    # Generate PDF report
    pdf_path = generate_report(input_dict, risk, err, penalty, shap_df)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name="patient_report.pdf",
            mime="application/pdf",
        )
