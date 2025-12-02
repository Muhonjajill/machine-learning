#!/usr/bin/env python

import pandas as pd
import pyodbc
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
import os
import numpy as np
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# DATABASE CONNECTION
# =========================================================
server = os.getenv("DB_SERVER")
database = os.getenv("DB_DATABASE")
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")

connection_string = f"""
DRIVER={{ODBC Driver 17 for SQL Server}};
SERVER={server};
DATABASE={database};
UID={username};
PWD={password};
TrustServerCertificate=yes;
"""

try:
    conn = pyodbc.connect(connection_string)
    print("Connected to SQL Server successfully!")
except Exception as e:
    print("Database connection failed:", e)
    raise


# =========================================================
# FETCH TRANSACTION + TERMINAL DATA
# =========================================================
query = """
SELECT 
    t.terminalid,
    t.name AS terminal_name,
    t.location,
    tr.trandate,
    tr.cashin
FROM [transactions] AS tr
JOIN [terminals] AS t
    ON tr.terminalid = t.terminalid
WHERE 
    tr.trandate >= '2025-01-01'

    AND tr.cashin IS NOT NULL
"""

df = pd.read_sql(query, conn)   
df['trandate'] = pd.to_datetime(df['trandate'])
df['cashin'] = pd.to_numeric(df['cashin'], errors='coerce')

df = df.groupby(['terminalid', 'terminal_name', 'location', 
                 pd.Grouper(key='trandate', freq='D')], 
                dropna=False).agg({'cashin': 'sum'}).reset_index()

terminals = df['terminalid'].unique()
print(f"Retrieved data for {len(terminals)} terminals (joined with terminals info)")


# =========================================================
# FORECAST WITH PROPHET PER TERMINAL
# =========================================================
output_dir = "forecasts"
os.makedirs(output_dir, exist_ok=True)
all_forecasts = []

for terminal in terminals:
    terminal_data = df[df['terminalid'] == terminal].dropna(subset=['cashin'])

    if len(terminal_data) < 3:
        print(f"Skipping {terminal} (not enough data)")
        continue

    print(f"\nForecasting cashin for terminal: {terminal}")

    data = (terminal_data
            .groupby(['trandate'])['cashin']
            .sum()
            .reset_index()
            .rename(columns={'trandate': 'ds', 'cashin': 'y'}))
    
    data = data.drop_duplicates(subset=['ds'], keep='last')
    data = data.sort_values('ds')

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode='additive',
        interval_width=0.80,  
        uncertainty_samples=1000  
    )
    model.fit(data)

    future = model.make_future_dataframe(periods=10)
    forecast = model.predict(future)

    # ---------------------------------------------------------
    # MODEL CONFIDENCE CALCULATION
    # ---------------------------------------------------------
    forecast['uncertainty_width'] = forecast['yhat_upper'] - forecast['yhat_lower']

    mean_yhat = forecast['yhat'].replace(0, np.nan).mean()
    forecast['confidence_score'] = 1 - (forecast['uncertainty_width'] / mean_yhat)
    forecast['confidence_score'] = forecast['confidence_score'].clip(0, 1)

    terminal_conf = forecast['confidence_score'].mean()
    print(f"Confidence for {terminal}: {terminal_conf:.2f}")

    historical_actuals = data['y'].values
    historical_predictions = model.predict(data)['yhat'].values
    prediction_errors = historical_actuals - historical_predictions

    mae = np.abs(prediction_errors).mean()
    std_error = np.std(prediction_errors)

    forecast['yhat_lower'] = forecast['yhat'] - (1.5 * std_error)
    forecast['yhat_upper'] = forecast['yhat'] + (1.5 * std_error)

    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    mean_historical = data['y'].mean()
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=mean_historical * 3)

    forecast['terminalid'] = terminal
    info = terminal_data[['terminal_name', 'location']].iloc[0].to_dict()
    forecast['terminal_name'] = info.get('terminal_name', '')
    forecast['location'] = info.get('location', '')

    plt.figure(figsize=(10, 5))
    model.plot(forecast)
    plt.title(f"Cashin Forecast - {terminal}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{terminal}_cashin_forecast.png")
    plt.close()

    simplified = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper',
                           'terminalid', 'terminal_name', 'location']]
    all_forecasts.append(simplified)


# ---------------------------------------------------------
# Optional Plotly visualization
# ---------------------------------------------------------
all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
terminal_to_plot = all_forecasts_df['terminalid'].unique()[0]
plot_df = all_forecasts_df[all_forecasts_df['terminalid'] == terminal_to_plot]

fig = px.line(
    plot_df,
    x='ds',
    y=['yhat', 'yhat_lower', 'yhat_upper'],
    title=f"Cash-in Forecast for Terminal {terminal_to_plot}",
    labels={'ds': 'Date', 'value': 'Forecast (KES)'},
)
fig.show()


# =========================================================
# SAVE FORECASTS & FETCH ACTUAL CASH-IN
# =========================================================
if all_forecasts:
    csv_path = f"{output_dir}/cashin_forecasts.csv"
    all_forecasts_df.to_csv(csv_path, index=False)
    print(f"Saved combined forecast to: {csv_path}")
else:
    print("No forecasts generated.")

print("Fetching REAL actual cash-in data...")

start_date = all_forecasts_df['ds'].min().strftime('%Y-%m-%d')

actual_query = """
SELECT 
    terminalid,
    CAST(trandate AS DATE) AS trandate,
    SUM(cashin) AS actual_cashin
FROM transactions
WHERE trandate >= ?
GROUP BY terminalid, CAST(trandate AS DATE)
"""

actual_df = pd.read_sql(actual_query, conn, params=[start_date])


# =========================================================
# MERGE FORECAST + ACTUALS + ANOMALY DETECTION
# =========================================================
forecast_df = all_forecasts_df.copy()
forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.date
actual_df['trandate'] = pd.to_datetime(actual_df['trandate']).dt.date

merged = pd.merge(
    forecast_df,
    actual_df,
    left_on=['terminalid', 'ds'],
    right_on=['terminalid', 'trandate'],
    how='left'
)

merged['actual'] = merged['actual_cashin']
merged['difference'] = merged['actual_cashin'] - merged['yhat']
merged['percent_diff'] = (merged['difference'] / merged['yhat']) * 100


# ---------- ROLLING Z-SCORE ----------
merged = merged.sort_values(by=['terminalid', 'ds'])

merged['roll_mean'] = merged.groupby('terminalid')['actual_cashin'].transform(
    lambda x: x.rolling(window=14, min_periods=3).mean()
)
merged['roll_std'] = merged.groupby('terminalid')['actual_cashin'].transform(
    lambda x: x.rolling(window=14, min_periods=3).std()
)

merged['roll_std'] = merged['roll_std'].replace(0, np.nan)
merged['zscore'] = (merged['actual_cashin'] - merged['roll_mean']) / merged['roll_std']


def zscore_flag(z):
    if pd.isna(z):
        return "NORMAL"
    if z > 2.5:
        return "SPIKE"
    if z < -2.5:
        return "DROP"
    return "NORMAL"


merged['z_anomaly'] = merged['zscore'].apply(zscore_flag)


merged['anomaly'] = merged.apply(
    lambda r: "HIGH" if r.actual_cashin > r.yhat_upper else
              "LOW" if r.actual_cashin < r.yhat_lower else
              "NORMAL",
    axis=1
)


def final_flag(row):
    prophet_flag = row['anomaly']
    z_flag = row['z_anomaly']

    if prophet_flag == "HIGH" and z_flag == "SPIKE":
        return "HIGH SPIKE"
    if prophet_flag == "LOW" and z_flag == "DROP":
        return "LOW DROP"
    if prophet_flag == "HIGH" and row['zscore'] > 2:
        return "HIGH (Z)"
    if prophet_flag == "LOW" and row['zscore'] < -2:
        return "LOW (Z)"
    return "NORMAL"


merged['final_anomaly'] = merged.apply(final_flag, axis=1)

print("Merged forecast + actual data with anomaly scoring")


# =========================================================
# PREPARE DATA FOR BACKEND
# =========================================================
export_df = merged.copy()

date_cols = ['ds', 'trandate']
for col in date_cols:
    if col in export_df.columns:
        export_df[col] = export_df[col].astype(str)

numeric_cols = ['difference', 'percent_diff', 'actual_cashin', 'actual',
                'yhat', 'yhat_lower', 'yhat_upper', 'roll_mean',
                'roll_std', 'zscore']

for col in numeric_cols:
    if col in export_df.columns:
        export_df[col] = pd.to_numeric(export_df[col], errors='coerce')
        export_df[col] = export_df[col].replace([np.inf, -np.inf], np.nan)
        export_df[col] = export_df[col].fillna(0)

export_df = export_df.where(pd.notnull(export_df), None)

print("Data cleaned for JSON serialization (guaranteed)")



# =========================================================
# SEND TO BACKEND API
# =========================================================

try:
    backend_url = os.getenv("BACKEND_URL") + "/api/cashForecasts"
    print(f"Sending forecast data to backend: {backend_url}")

    payload = export_df.to_dict(orient='records')

    response = requests.post(backend_url, json=payload)

    if response.status_code == 200:
        print("Forecast data sent successfully to backend!")
    else:
        print(f"Failed to send data — {response.status_code}: {response.text}")

except Exception as e:
    print(f"Error sending data to backend: {e}")