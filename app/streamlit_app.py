import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import streamlit as st
import altair as alt

from src.etl import transform
from src.predict import predict_sample
from src.explain import explain_sample
from src.finance import calculate_penalty
from src.report import generate_report

# --- Page Configuration ---
st.set_page_config(
    page_title="PenaltyGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit marks */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom padding */
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Metric Cards */
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    text-align: center;
    border-top: 5px solid #1f77b4;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 20px;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
}
.metric-title {
    font-size: 0.9rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.2;
}

/* Colors for contexts */
.border-low { border-top-color: #10b981 !important; }
.border-medium { border-top-color: #f59e0b !important; }
.border-high { border-top-color: #ef4444 !important; }
.border-finance { border-top-color: #6366f1 !important; }

.text-low { color: #10b981 !important; }
.text-medium { color: #f59e0b !important; }
.text-high { color: #ef4444 !important; }
.text-finance { color: #334155 !important; }

/* Dark mode compatibility */
@media (prefers-color-scheme: dark) {
    .metric-card {
        background-color: #1e293b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title { color: #94a3b8; }
    .metric-value { color: #f8fafc; }
    .text-finance { color: #f8fafc !important; }
}

/* Headers */
.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.sub-header {
    font-size: 1.2rem;
    color: #64748b;
    margin-top: 0px;
    margin-bottom: 2rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="main-header">PenaltyGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Readmission Risk & Financial Penalty Forecasting</div>', unsafe_allow_html=True)
st.markdown("---")

# --- Input Form ---
with st.form("patient_form"):
    st.markdown("### 📋 Patient Assessment Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Demographics & Region**")
        age = st.number_input("Age", 0, 120, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Central", "East", "North", "South", "West"])
        insurance_type = st.selectbox("Insurance Type", ["Medicaid", "Medicare", "Private", "Uninsured"])

    with col2:
        st.markdown("**2. Clinical History**")
        primary_diagnosis = st.selectbox("Primary Diagnosis", [
            "Appendicitis", "COPD", "Diabetes", "Fracture", "Heart Failure",
            "Hypertension", "Influenza", "Kidney Disease", "Pneumonia",
            "Sepsis", "Stroke"
        ])
        comorbidities_count = st.number_input("Comorbidities Count", 0, 10, 2)
        prev_readmissions = st.number_input("Previous Readmissions", 0, 10, 0)
        treatment_type = st.selectbox("Treatment Type", [
            "Conservative", "Interventional", "Medical", "Surgical"
        ])

    with col3:
        st.markdown("**3. Encounter Details**")
        season = st.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])
        length_of_stay = st.number_input("Length of Stay (Days)", 1, 30, 5)
        medications_count = st.number_input("Medications Count", 0, 20, 6)
        followup_visits_last_year = st.number_input("Followups Last Year", 0, 20, 1)
        discharge_disposition = st.selectbox("Discharge Disposition", [
            "Home", "Home Health", "Rehab", "Skilled Nursing"
        ])
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**4. Financial Parameters**")
    revenue = st.number_input("Hospital Annual Revenue Basis (₹)", value=10000000, step=1000000, format="%d")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Calculate Risk & Financial Impact", use_container_width=True)

# --- Results Section ---
if submitted:
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

    # Transform and Predict
    input_df = pd.DataFrame([input_dict])
    input_encoded = transform(input_df)
    risk = predict_sample(input_encoded)

    # Determine styles based on risk
    if risk < 0.2:
        risk_level = "Low"
        color_class = "border-low text-low"
    elif risk < 0.5:
        risk_level = "Medium"
        color_class = "border-medium text-medium"
    else:
        risk_level = "High"
        color_class = "border-high text-high"

    # Financial & SHAP
    err, penalty = calculate_penalty(risk, revenue)
    shap_df = explain_sample(input_encoded)

    st.markdown("---")
    st.markdown("## 📊 Assessment Results")
    
    # 4-Column Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card {color_class}">
            <div class="metric-title">Risk Score</div>
            <div class="metric-value {color_class}">{risk:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card {color_class}">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value {color_class}">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-card border-finance">
            <div class="metric-title">Expected Readmission Ratio (ERR)</div>
            <div class="metric-value text-finance">{err:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card border-high">
            <div class="metric-title">Estimated Penalty Found</div>
            <div class="metric-value text-high">₹{penalty:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Top Risk Drivers")

    # Altair SHAP Chart
    shap_df["color"] = shap_df["shap_value"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")
    shap_df["Impact"] = shap_df["shap_value"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
    
    chart_df = shap_df.sort_values(by="shap_value", ascending=True)

    chart = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        x=alt.X("shap_value:Q", title="SHAP Value (Impact on Prediction)"),
        y=alt.Y("feature:N", sort=alt.EncodingSortField(field="shap_value", order="ascending"), title="Patient Feature"),
        color=alt.Color("color:N", scale=None),
        tooltip=[
            alt.Tooltip("feature", title="Feature"),
            alt.Tooltip("shap_value", title="SHAP Value", format=".3f"),
            alt.Tooltip("Impact", title="Driver")
        ]
    ).properties(height=350).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        grid=True,
        gridOpacity=0.2
    ).configure_view(strokeWidth=0)

    st.altair_chart(chart, use_container_width=True)

    # Export Report
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_path = generate_report(input_dict, risk, err, penalty, shap_df)
    
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Comprehensive PDF Report",
            data=f,
            file_name="patient_report.pdf",
            mime="application/pdf",
            type="primary"
        )
