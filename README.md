# PenaltyGuard AI

A machine learning application for hospital readmission prediction and penalty risk analysis.

## Project Structure

```
PenaltyGuard-AI/
├── data/                  # Dataset directory
├── src/
│   ├── etl.py             # Data extraction, transformation, loading
│   ├── train.py           # Model training
│   ├── predict.py         # Predictions
│   ├── explain.py         # Model explainability (SHAP)
│   └── finance.py         # Financial impact analysis
├── app/
│   └── streamlit_app.py   # Streamlit dashboard
├── models/                # Saved model artifacts
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app/streamlit_app.py
```
