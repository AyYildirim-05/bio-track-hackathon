# PEPTIDE-SCAN 2000 — Anticancer Peptide (ACP) Predictor

## Quickstart

### 1. Generate Baseline Model (or Train on Custom CSV)
```bash
# Train quick baseline model (saves acp_model.joblib)
python baseline.py

# Or train the full RF + LightGBM ensemble with 5-fold CV on a CSV dataset
python train.py --data path/to/dataset.csv --seq-col sequence --label-col label --out acp_model.joblib
```

### 2. Launch the Streamlit UI
```bash
streamlit run app.py
```
