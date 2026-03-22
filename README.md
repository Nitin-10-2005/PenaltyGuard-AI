# PenaltyGuard AI

A machine learning application for hospital readmission prediction and penalty risk analysis.

## Project Structure

```
PenaltyGuard-AI/
├── data/                  # Dataset directory
├── src/
│   ├── etl.py             # Data extraction, transformation, loading
│   ├── train.py           # Model training (XGBoost + GridSearchCV)
│   ├── predict.py         # Predictions
│   ├── explain.py         # Model explainability (SHAP)
│   ├── finance.py         # Financial impact analysis
│   └── report.py          # PDF report generation
├── app/
│   └── streamlit_app.py   # Streamlit dashboard
├── models/                # Saved model artifacts
├── reports/               # Generated PDF reports
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Train the model
```bash
python src/train.py
```

### 2. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

### 3. Run individual modules
```bash
python src/etl.py        # Test ETL pipeline
python src/predict.py    # Test prediction
python src/explain.py    # Test SHAP explanations
python src/finance.py    # Test penalty calculations
python src/report.py     # Generate sample PDF report
```
