#!/usr/bin/env python

import pandas as pd
import pyodbc
from prophet import Prophet
import matplotlib.pyplot as plt
import requests
import os
import numpy as np



server = '41.207.70.18,60001'   # include port with comma
database = 'brits_ncba'
username = 'brits_ncba'
password = '6fuAFMCeQvtk7L8wn5sl'

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
    print("✅ Connected to SQL Server successfully!")
except Exception as e:
    print("❌ Database connection failed:", e)
    raise


# In[25]:


# 🧩 Cell 3 — Fetch Data with Terminal Info
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
    tr.trandate >= DATEADD(DAY, -180, GETDATE())
    AND tr.cashin IS NOT NULL
"""

conn = pyodbc.connect(connection_string)
df = pd.read_sql(query, conn)
conn.close()

# Clean
df['trandate'] = pd.to_datetime(df['trandate'])
df['cashin'] = pd.to_numeric(df['cashin'], errors='coerce')

terminals = df['terminalid'].unique()
print(f"✅ Retrieved data for {len(terminals)} terminals (joined with terminals info)")
df.head()


# In[26]:


output_dir = "forecasts"
os.makedirs(output_dir, exist_ok=True)
all_forecasts = []

for terminal in terminals:
    terminal_data = df[df['terminalid'] == terminal].dropna(subset=['cashin'])

    if len(terminal_data) < 3:
        print(f"⚠️ Skipping {terminal} (not enough data)")
        continue

    print(f"\n🔮 Forecasting cashin for terminal: {terminal}")

    data = terminal_data.rename(columns={'trandate': 'ds', 'cashin': 'y'})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(data)

    # Predict next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    # Ensure forecasts never go below 0
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    forecast['terminalid'] = terminal

    # Add terminal name and location if available
    terminal_info = terminal_data[['terminal_name', 'location']].iloc[0].to_dict()
    forecast['terminal_name'] = terminal_info.get('terminal_name', '')
    forecast['location'] = terminal_info.get('location', '')

    # Save chart
    plt.figure(figsize=(10, 5))
    model.plot(forecast)
    plt.title(f"Cashin Forecast - {terminal}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{terminal}_cashin_forecast.png")
    plt.close()

    simplified = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'terminalid', 'terminal_name', 'location']]
    all_forecasts.append(simplified)


# In[27]:


import pandas as pd
import plotly.express as px

# Combine all forecasts into one DataFrame
all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)

# Example: visualize forecast for a single terminal
terminal_to_plot = all_forecasts_df['terminalid'].unique()[0]  # change if you want another
plot_df = all_forecasts_df[all_forecasts_df['terminalid'] == terminal_to_plot]

fig = px.line(
    plot_df,
    x='ds',
    y=['yhat', 'yhat_lower', 'yhat_upper'],
    title=f"📈 Cash-in Forecast for Terminal {terminal_to_plot}",
    labels={'ds': 'Date', 'value': 'Forecast (KES)'},
)
fig.show()


# In[28]:


# 🧩 Cell 5 — Combine Forecasts
if all_forecasts:
    all_forecasts_df = pd.concat(all_forecasts)
    csv_path = f"{output_dir}/cashin_forecasts.csv"
    all_forecasts_df.to_csv(csv_path, index=False)
    print(f"📂 Saved combined forecast to: {csv_path}")
else:
    print("⚠️ No forecasts generated.")

# AFTER forecast generation – merge with ACTUALS


print("🔄 Fetching REAL actual cash-in data...")
# Get the first forecast date
start_date = all_forecasts_df['ds'].min().strftime('%Y-%m-%d')
actual_query = """
SELECT 
    terminalid,
    CAST(trandate AS DATE) AS trandate,
    SUM(cashin) AS actual_cashin
FROM transactions
WHERE trandate >=  ?
GROUP BY terminalid, CAST(trandate AS DATE)
"""
conn = pyodbc.connect(connection_string)
actual_df = pd.read_sql(actual_query, conn, params=[start_date])
conn.close()

#actual_df['trandate'] = pd.to_datetime(actual_df['trandate'])

# Merge Actual + Forecast
forecast_df = all_forecasts_df.copy()
#forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.date
actual_df['trandate'] = pd.to_datetime(actual_df['trandate']).dt.date
merged = pd.merge(
    forecast_df,
    actual_df,
    left_on=['terminalid', 'ds'],
    right_on=['terminalid', 'trandate'],
    how='left'
)
merged['actual'] = merged['actual_cashin']   # DO NOT fill with 0
# Compute anomaly score
merged['difference'] = merged['actual_cashin'] - merged['yhat']
merged['percent_diff'] = (merged['difference'] / merged['yhat']) * 100

merged['anomaly'] = merged.apply(
    lambda r: "HIGH" if r.actual_cashin > r.yhat_upper else
              "LOW" if r.actual_cashin < r.yhat_lower else
              "NORMAL",
    axis=1
)

print("✅ Merged forecast + actual data with anomaly scoring")

# Prepare final export dataset
export_df = merged.copy()
export_df['ds'] = export_df['ds'].astype(str)

# Replace NaN in 'difference' and 'percent_diff' only, not actual_cashin
export_df['difference'] = export_df['difference'].fillna(0)
export_df['percent_diff'] = export_df['percent_diff'].fillna(0)

# Do NOT fill 'actual_cashin'
#export_df = export_df.rename(columns={'actual_cashin': 'actual'})



# In[29]:


import requests

try:
    backend_url = "http://localhost:5000/api/cashForecasts"  # change port if needed
    print(f"🚀 Sending forecast data to backend: {backend_url}")

    for col in export_df.columns:
        if pd.api.types.is_datetime64_any_dtype(export_df[col]):
            export_df[col] = export_df[col].astype(str)

    # Also force-convert date columns (Python date objects)
    export_df['ds'] = export_df['ds'].astype(str)
    if 'trandate' in export_df.columns:
        export_df['trandate'] = export_df['trandate'].astype(str)
    # Remove NaT and NaN so JSON can serialize safely
    export_df = export_df.replace([np.nan, np.inf, -np.inf], None)
    export_df = export_df.replace("NaT", None)


    # Convert datetime columns (like 'ds') to string (ISO format)
    """all_forecasts_df['ds'] = all_forecasts_df['ds'].astype(str)

    # Convert Decimal or float64 columns just in case
    all_forecasts_df = all_forecasts_df.astype({
        'yhat': 'float',
        'yhat_lower': 'float',
        'yhat_upper': 'float'
    })"""

    #payload = all_forecasts_df.to_dict(orient='records')
    payload = export_df.to_dict(orient='records')

    response = requests.post(backend_url, json=payload)

    if response.status_code == 200:
        print("✅ Forecast data sent successfully to backend!")
    else:
        print(f"❌ Failed to send data — {response.status_code}: {response.text}")

except Exception as e:
    print(f"⚠️ Error sending data to backend: {e}")







