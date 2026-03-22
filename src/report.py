"""Report: Generate PDF patient report using reportlab."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register a Unicode font that supports ₹
_UNICODE_FONT = "Helvetica"  # fallback
_UNICODE_FONT_FILE = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
if os.path.exists(_UNICODE_FONT_FILE):
    pdfmetrics.registerFont(TTFont("ArialUnicode", _UNICODE_FONT_FILE))
    _UNICODE_FONT = "ArialUnicode"

_RUPEE = "INR "


def generate_report(input_data, risk, err, penalty, shap_df):
    """Generate a PDF report and save to reports/ folder."""

    os.makedirs("reports", exist_ok=True)
    filepath = "reports/patient_report.pdf"

    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()

    # Override default fonts with Unicode-capable font
    for style_name in styles.byName:
        styles[style_name].fontName = _UNICODE_FONT
    styles["Title"].fontName = _UNICODE_FONT
    styles["Heading2"].fontName = _UNICODE_FONT
    elements = []

    # Title
    elements.append(Paragraph("PenaltyGuard AI Report", styles["Title"]))
    elements.append(Spacer(1, 0.5 * cm))

    # Patient Details
    elements.append(Paragraph("Patient Details", styles["Heading2"]))
    patient_data = [[str(k), str(v)] for k, v in input_data.items()]
    patient_table = Table(patient_data, colWidths=[7 * cm, 9 * cm])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (0, -1), _UNICODE_FONT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Risk Score & Level
    elements.append(Paragraph("Risk Assessment", styles["Heading2"]))

    if risk < 0.2:
        risk_level = "Low"
    elif risk < 0.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    risk_data = [
        ["Risk Score", f"{risk:.4f}"],
        ["Risk Level", risk_level],
    ]
    risk_table = Table(risk_data, colWidths=[7 * cm, 9 * cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (0, -1), _UNICODE_FONT),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Financial Impact
    elements.append(Paragraph("Financial Impact", styles["Heading2"]))
    finance_data = [
        ["ERR", f"{err:.2f}"],
        ["Estimated Penalty", f"{_RUPEE}{penalty:,.2f}"],
    ]
    finance_table = Table(finance_data, colWidths=[7 * cm, 9 * cm])
    finance_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), _UNICODE_FONT),
    ]))
    elements.append(finance_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Top 5 SHAP Factors
    elements.append(Paragraph("Top 5 Risk Factors (SHAP)", styles["Heading2"]))
    top5 = shap_df.head(5)
    shap_data = [["Feature", "SHAP Value"]]
    for _, row in top5.iterrows():
        shap_data.append([str(row["feature"]), f"{row['shap_value']:.4f}"])

    shap_table = Table(shap_data, colWidths=[7 * cm, 9 * cm])
    shap_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), _UNICODE_FONT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ]))
    elements.append(shap_table)

    # Build PDF
    doc.build(elements)
    return filepath


if __name__ == "__main__":
    # Quick test with sample data
    sample_input = {
        "age": 65, "gender": "Male", "season": "Winter",
        "region": "Urban", "comorbidities_count": 3,
        "length_of_stay": 7, "medications_count": 8,
        "followup_visits_last_year": 2, "prev_readmissions": 1,
        "primary_diagnosis": "Cardiovascular", "treatment_type": "Surgical",
        "insurance_type": "Private", "discharge_disposition": "Home",
    }

    import pandas as pd
    sample_shap = pd.DataFrame({
        "feature": ["age", "comorbidities_count", "prev_readmissions", "length_of_stay", "medications_count"],
        "shap_value": [0.92, 0.50, -0.29, -0.28, 0.46],
        "abs_val": [0.92, 0.50, 0.29, 0.28, 0.46],
    })

    path = generate_report(sample_input, risk=0.80, err=1.67, penalty=300000, shap_df=sample_shap)
    print(f"Report saved to: {path}")
