import streamlit as st
import pandas as pd
import joblib, os
import plotly.express as px
from datetime import datetime, timedelta

st.title('AI Business Forecast Dashboard')

st.markdown('Upload a CSV with columns: date (YYYY-MM-DD), value')

uploaded = st.file_uploader('Upload CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['date'])
else:
    st.markdown('Using sample dataset.')
    df = pd.read_csv('data/monthly_kpis.csv', parse_dates=['date'])

st.dataframe(df.tail(12))

if st.button('Train model (local)'):
    st.info('Training model... run `python train.py` in terminal if you prefer.')
    os.system('python train.py')

if st.button('Load model and forecast next 6 months'):
    try:
        model = joblib.load('models/forecast_model.joblib')
    except Exception as e:
        st.error('Model not found. Run python train.py first.')
        st.stop()

    # prepare features for forecasting
    df = df.sort_values('date').reset_index(drop=True)
    df['value_lag1'] = df['value'].shift(1)
    df['value_lag2'] = df['value'].shift(2)
    df['month'] = df['date'].dt.month
    df = df.dropna().reset_index(drop=True)

    last_date = df['date'].max()
    future = []
    last_vals = list(df['value'].values[-2:])
    for i in range(6):
        next_month = (last_date + pd.DateOffset(months= i+1)).replace(day=1)
        vlag1 = last_vals[-1]
        vlag2 = last_vals[-2]
        month = next_month.month
        feat = [[vlag1, vlag2, month]]
        pred = model.predict(feat)[0]
        future.append({'date': next_month, 'value': pred})
        last_vals.append(pred)

    fut_df = pd.DataFrame(future)
    combined = pd.concat([df[['date','value']], fut_df], ignore_index=True)
    fig = px.line(combined, x='date', y='value', title='Historical + Forecast')
    st.plotly_chart(fig)
    st.dataframe(fut_df)

st.markdown('Basic anomaly detection: flag points with large drops compared to lag')
df['pct_change'] = df['value'].pct_change()
anoms = df[df['pct_change'] < -0.2]
if not anoms.empty:
    st.markdown('**Detected anomalies (drop >20%)**')
    st.dataframe(anoms[['date','value','pct_change']])
