# Integrated AI Supply Chain Bottleneck Predictor

## Workflow
The **Shipment analysis** page uses one form and runs:
1. Isolation Forest anomaly detection
2. One-Class SVM anomaly detection
3. Random Forest / XGBoost disruption classification

The **Model evaluation** and **Leakage audit** pages remain separate for reporting. Anomaly flags are not proof of disruption; classification still runs and the dashboard recommends human review.

## Run in VS Code (Windows PowerShell)
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app trains from `data/global_supply_chain_risk.csv` when it first starts. Metrics are dataset-specific; leakage audit reduces risk but does not prove zero leakage.
