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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Premium Dark SaaS CSS ---
custom_css = """
<style>
/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Base styling */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

/* Force dark background on Streamlit main app container */
.stApp {
    background-color: #0b0f19;
    background-image: radial-gradient(circle at 50% -20%, #1e1b4b, #0b0f19 60%);
}

/* Hide default streamlit marks */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom padding */
.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

/* SaaS Header Styling */
.saas-header {
    text-align: center;
    margin-bottom: 3.5rem;
    padding-top: 1rem;
}
.saas-logo {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.saas-title {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1rem;
    background: linear-gradient(to right, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.saas-subtitle {
    font-size: 1.125rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
    font-weight: 400;
}

/* Input Form Styling */
[data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* Form Section Headers */
.form-section-title {
    color: #f8fafc;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

/* Make inputs look integrated */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
    transition: all 0.2s ease;
}
.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}
.stMarkdown p {
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* CTA Button */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
    width: 100%;
    margin-top: 1rem;
}
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.5);
}

/* Glassmorphism Metric Cards */
.saas-metric-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    height: 100%;
}
.saas-metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px -10px rgba(0,0,0,0.6);
    background: rgba(30, 41, 59, 0.6);
    border-color: rgba(255, 255, 255, 0.1);
}

/* Glow effects based on state */
.glow-high::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #ef4444, transparent); opacity: 0.8;
}
.glow-medium::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #f59e0b, transparent); opacity: 0.8;
}
.glow-low::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #10b981, transparent); opacity: 0.8;
}
.glow-finance::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #6366f1, transparent); opacity: 0.8;
}

.saas-metric-title {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.saas-metric-value {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.25rem;
    letter-spacing: -1px;
}

/* Color classes */
.text-high { color: #fca5a5; text-shadow: 0 0 30px rgba(239, 68, 68, 0.3); }
.text-medium { color: #fcd34d; text-shadow: 0 0 30px rgba(245, 158, 11, 0.3); }
.text-low { color: #86efac; text-shadow: 0 0 30px rgba(16, 185, 129, 0.3); }
.text-finance { color: #f8fafc; text-shadow: 0 0 30px rgba(255, 255, 255, 0.2); }

/* Section Headers */
.section-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #ffffff;
    margin: 4rem 0 2rem 0;
    letter-spacing: -0.5px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- App Structure ---
st.markdown("""
<div class="saas-header">
    <div class="saas-logo">✨ AI Healthcare Engine</div>
    <div class="saas-title">PenaltyGuard AI</div>
    <div class="saas-subtitle">Predict hospital readmission risks and forecast CMS financial penalties with real-time explainable AI.</div>
</div>
""", unsafe_allow_html=True)

with st.form("patient_form"):
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="form-section-title">👤 Demographics & Profile</div>', unsafe_allow_html=True)
        age = st.number_input("Age", 0, 120, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Central", "East", "North", "South", "West"])
        insurance_type = st.selectbox("Insurance Type", ["Medicaid", "Medicare", "Private", "Uninsured"])

    with col2:
        st.markdown('<div class="form-section-title">🏥 Clinical History</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="form-section-title">⏱️ Encounter Details</div>', unsafe_allow_html=True)
        season = st.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])
        length_of_stay = st.number_input("Length of Stay (Days)", 1, 30, 5)
        medications_count = st.number_input("Medications Count", 0, 20, 6)
        followup_visits_last_year = st.number_input("Followups Last Year", 0, 20, 1)
        discharge_disposition = st.selectbox("Discharge Disposition", [
            "Home", "Home Health", "Rehab", "Skilled Nursing"
        ])
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="form-section-title">💰 Financial Baseline</div>', unsafe_allow_html=True)
    revenue = st.number_input("Hospital Annual Revenue Basis (₹)", value=10000000, step=1000000, format="%d")

    # The use_container_width deprecation notice you saw earlier is addressed by using width='stretch', but button doesn't support width arg yet in some versions, so use_container_width=True is still fine, just prints a warning.
    submitted = st.form_submit_button("Run Analytics Engine", use_container_width=True)

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
        glow_class = "glow-low"
        text_class = "text-low"
    elif risk < 0.5:
        risk_level = "Medium"
        glow_class = "glow-medium"
        text_class = "text-medium"
    else:
        risk_level = "High"
        glow_class = "glow-high"
        text_class = "text-high"

    # Financial & SHAP
    err, penalty = calculate_penalty(risk, revenue)
    shap_df = explain_sample(input_encoded)

    st.markdown('<div class="section-title">📊 Assessment Results</div>', unsafe_allow_html=True)
    
    # 4-Column Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="saas-metric-card {glow_class}">
            <div class="saas-metric-title">Risk Score</div>
            <div class="saas-metric-value {text_class}">{risk:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="saas-metric-card {glow_class}">
            <div class="saas-metric-title">Risk Severity</div>
            <div class="saas-metric-value {text_class}">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="saas-metric-card glow-finance">
            <div class="saas-metric-title">Expected Readmission Ratio</div>
            <div class="saas-metric-value text-finance">{err:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="saas-metric-card glow-high">
            <div class="saas-metric-title">Estimated Penalty</div>
            <div class="saas-metric-value text-high">₹{penalty:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔍 Feature Impact Analysis</div>', unsafe_allow_html=True)

    # Altair SHAP Chart - Dark Theme
    shap_df["color"] = shap_df["shap_value"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")
    shap_df["Impact"] = shap_df["shap_value"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
    
    chart_df = shap_df.sort_values(by="shap_value", ascending=True)

    chart = alt.Chart(chart_df).mark_bar(
        cornerRadiusEnd=6, 
        cornerRadiusTopRight=6, 
        cornerRadiusBottomRight=6,
        size=16
    ).encode(
        x=alt.X("shap_value:Q", title="SHAP Value (Impact)", axis=alt.Axis(grid=False)),
        y=alt.Y("feature:N", sort=alt.EncodingSortField(field="shap_value", order="ascending"), title=""),
        color=alt.Color("color:N", scale=None),
        tooltip=[
            alt.Tooltip("feature", title="Feature"),
            alt.Tooltip("shap_value", title="SHAP Value", format=".3f"),
            alt.Tooltip("Impact", title="Driver")
        ]
    ).properties(height=380).configure_axis(
        labelFontSize=13,
        titleFontSize=14,
        labelColor="#94a3b8",
        titleColor="#94a3b8",
        grid=False,
        domainColor="#334155",
        tickColor="#334155"
    ).configure_view(strokeWidth=0).configure(background="transparent")

    # Create a nice container for the chart
    st.markdown("""
    <style>
    .chart-container {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 2rem;
        backdrop-filter: blur(20px);
    }
    </style>
    <div class="chart-container">
    """, unsafe_allow_html=True)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Export Report
    st.markdown("<br><br>", unsafe_allow_html=True)
    pdf_path = generate_report(input_dict, risk, err, penalty, shap_df)
    
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Comprehensive PDF Report",
            data=f,
            file_name="patient_report.pdf",
            mime="application/pdf",
            type="primary"
        )
