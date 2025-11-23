import pandas as pd
import numpy as np
import os, joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/monthly_kpis.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Basic feature engineering: lag features
df['value_lag1'] = df['value'].shift(1)
df['value_lag2'] = df['value'].shift(2)
df['month'] = df['date'].dt.month
df = df.dropna().reset_index(drop=True)

features = ['value_lag1','value_lag2','month']
X = df[features]
y = df['value']

# simple time-series CV
tscv = TimeSeriesSplit(n_splits=3)
errors = []
models = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    err = mean_absolute_error(y_test, pred)
    errors.append(err)
    models.append(model)

print('CV MAE scores:', errors)
# pick last model as production
os.makedirs('models', exist_ok=True)
joblib.dump(models[-1], 'models/forecast_model.joblib')
print('Saved models/forecast_model.joblib')
