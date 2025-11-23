# AI Business Forecast Dashboard

Predictive analytics dashboard for forecasting monthly KPIs (revenue, leads, or ticket volume).
Includes data cleaning, multiple model flavors (Prophet & XGBoost fallback), confidence intervals,
anomaly detection, and a Streamlit visual dashboard.

## Structure
- data/monthly_kpis.csv  (sample synthetic dataset)
- train.py                (trains models, saves models/forecast_model.joblib)
- app.py                  (Streamlit dashboard to upload CSV and forecast)
- requirements.txt
- Dockerfile
- LICENSE (MIT)

## Quick start
```bash
# create venv & install
pip install -r requirements.txt

# train (creates models/forecast_model.joblib)
python train.py

# run the dashboard
streamlit run app.py
```

## Notes
- The project includes a simple Prophet-like forecasting approach using statsmodels seasonal decomposition and XGBoost as a machine-learning fallback.
- Replace the sample CSV with your real KPI CSV with columns: date (YYYY-MM-DD), value.
