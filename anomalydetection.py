#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install dependencies if not already installed



# In[2]:


import pyodbc
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# In[12]:


from datetime import datetime, timedelta
import pyodbc
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


# -----------------------------
# 2. LOAD LAST 180 DAYS
# -----------------------------
START_DATE = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

sql_tx = f"""
SELECT 
    trandate, terminalid, cashin, custid, custname
FROM transactions
WHERE trandate >= '{START_DATE}'
  AND cashin > 0     -- only deposits matter
ORDER BY terminalid, trandate
"""

df = pd.read_sql(sql_tx, conn)
df['trandate'] = pd.to_datetime(df['trandate'])
#print(df.head())

#. Transaction-Level Feature Engineering
# Sort so rolling windows work correctly
df = df.sort_values(['custid', 'trandate']).reset_index(drop=True)

# Transaction level features
df['hour'] = df['trandate'].dt.hour
df['day_of_week'] = df['trandate'].dt.dayofweek   # 0 = Monday
df['day'] = df['trandate'].dt.date

#B. Aggregate Customer Daily Features
daily = df.groupby(['custid', 'custname', 'day']).agg(
    total_cashin = ('cashin', 'sum'),
    txn_count = ('cashin', 'count'),
    avg_txn_size = ('cashin', 'mean'),
    max_txn = ('cashin', 'max'),
    min_txn = ('cashin', 'min')
).reset_index()

# Convert day to datetime
daily['day'] = pd.to_datetime(daily['day'])

# C. Advanced Rolling Statistics Per Customer
daily = daily.sort_values(['custid', 'day'])

# Rolling 7-day window for each customer
daily['rolling_mean_7'] = daily.groupby('custid')['total_cashin'] \
    .transform(lambda s: s.rolling(7, min_periods=3).mean())

daily['rolling_std_7'] = daily.groupby('custid')['total_cashin'] \
    .transform(lambda s: s.rolling(7, min_periods=3).std())

daily['rolling_txn_count_7'] = daily.groupby('custid')['txn_count'] \
    .transform(lambda s: s.rolling(7, min_periods=3).mean())

#D. Z-Score Feature (Simple Anomaly Score)
daily['zscore'] = (daily['total_cashin'] - daily['rolling_mean_7']) / daily['rolling_std_7']

#E. Percent Change & Volatility
daily['pct_change'] = daily.groupby('custid')['total_cashin'].pct_change()
daily['volatility_7'] = daily.groupby('custid')['total_cashin'] \
    .transform(lambda s: s.rolling(7, min_periods=3).std())

#F. Behavior Frequency Features
daily['is_weekend'] = daily['day'].dt.dayofweek >= 5
daily['weekday_avg'] = daily.groupby(['custid', 'is_weekend'])['total_cashin'] \
    .transform('mean')

print(daily.head(10))
print(daily.columns)


# In[13]:


# Step 2 — LIGHTWEIGHT ANOMALY DETECTION (based on your engineered features)

# Replace infinite values from pct_change
daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)

# Simple Z-score flag
daily['anomaly_z'] = daily['zscore'].abs() > 3

print("Z-score anomalies:", daily['anomaly_z'].sum())


# In[15]:


#Step 3 — MACHINE LEARNING MODEL: IsolationForest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ML features
features = [
    'total_cashin', 'txn_count', 'avg_txn_size',
    'rolling_mean_7', 'rolling_std_7',
    'pct_change', 'volatility_7'
]

X = daily[features].fillna(0).values

# Standardize
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Global model
iso = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42
)

iso.fit(Xs)

daily['iso_flag'] = iso.predict(Xs) == -1
daily['iso_score'] = iso.decision_function(Xs)


# ----------------------------------------------------
# CONFIDENCE TESTING SECTION
# ----------------------------------------------------



print("\n🔍 Running model confidence evaluation...")

def stability_test(Xs, runs=5, sample_fraction=0.8):
    """Runs IsolationForest multiple times to check stability."""
    flags = []

    for i in range(runs):
        iso_tmp = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=42 + i
        )
        idx = np.random.choice(len(Xs), int(len(Xs) * sample_fraction), replace=False)
        iso_tmp.fit(Xs[idx])
        preds = (iso_tmp.predict(Xs) == -1).astype(int)
        flags.append(preds)

    flags = np.array(flags)
    pair_similarities = []

    for i in range(runs):
        for j in range(i + 1, runs):
            sim = np.mean(flags[i] == flags[j])
            pair_similarities.append(sim)

    return np.mean(pair_similarities)

stability_score = stability_test(Xs)
print(f"🔵 Model Stability Score: {stability_score:.3f}")


# Temporal train-test split (70/30)
split_day = daily['day'].quantile(0.7)

train = daily[daily['day'] <= split_day]
test = daily[daily['day'] > split_day]

X_train = scaler.fit_transform(train[features].fillna(0))
X_test = scaler.transform(test[features].fillna(0))

iso_tmp = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=123
)
iso_tmp.fit(X_train)

test_preds = iso_tmp.predict(X_test)
test_scores = iso_tmp.decision_function(X_test)

# Score separation metric
separation = np.mean(test_scores[test_preds == 1]) - np.mean(test_scores[test_preds == -1])
print(f"🟣 Out-of-Sample Score Separation: {separation:.4f}")


# Customer-level confidence: stability + score distribution
daily['score_abs'] = daily['iso_score'].abs()

cust_conf = daily.groupby(['custid', 'custname']).agg(
    mean_score=('score_abs', 'mean'),
    anomaly_rate=('iso_flag', 'mean')
).reset_index()

# Normalize mean_score
cust_conf['confidence'] = (
    cust_conf['mean_score'] / cust_conf['mean_score'].max()
).clip(0, 1)

print("\n🟢 CUSTOMER CONFIDENCE SUMMARY:")
print(cust_conf[['custid', 'custname', 'confidence']].head(10))

# Prepare anomalies payload (list of dicts)
anomalies = daily[daily['iso_flag'] == True].copy()
# Combine everything into 0–1 confidence score
sep_norm = np.tanh(separation)  # safe -1..1 mapping

global_confidence = (
    0.5 * stability_score +
    0.3 * sep_norm +
    0.2 * (1 - anomalies.shape[0] / len(daily))
)

global_confidence = float(np.clip(global_confidence, 0, 1))


print(f"\n🔥 FINAL SCRIPT CONFIDENCE SCORE: {global_confidence:.3f}")


# Prepare anomalies payload (list of dicts)
#anomalies = daily[daily['iso_flag'] == True].copy()

# Select columns to send (you can add more)
cols_to_send = [
    'custid', 'custname', 'day', 'total_cashin', 'txn_count',
    'avg_txn_size', 'rolling_mean_7', 'rolling_std_7',
    'zscore', 'pct_change', 'volatility_7', 'iso_score'
]

payload = anomalies[cols_to_send].to_dict(orient='records')
print(f"✅ Detected {len(payload)} anomalies via IsolationForest")




# In[18]:


import matplotlib.pyplot as plt

# Step 1 — Get top 10 customers with most anomalies
top10_custs = (
    daily[daily['iso_flag'] == True]
    .groupby(['custid', 'custname'])
    .size()
    .reset_index(name='anomaly_count')
    .sort_values('anomaly_count', ascending=False)
    .head(10)
)

# Step 2 — Loop through each customer and plot
for idx, row in top10_custs.iterrows():
    custid = row['custid']
    custname = row['custname']
    
    temp = daily[daily['custid'] == custid]
    
    plt.figure(figsize=(14,6))
    plt.plot(temp['day'], temp['total_cashin'], label="Cash-in")
    
    anomaly_rows = temp[temp['iso_flag'] == True]
    plt.scatter(
        anomaly_rows['day'], anomaly_rows['total_cashin'],
        s=80, edgecolors='black', marker='o', label='Anomaly'
    )
    
    plt.title(f"Customer {custname} ({custid}) — Anomalies Detected")
    plt.xlabel("Date")
    plt.ylabel("Cash-in Amount")
    plt.legend()
    plt.show()


# In[19]:


plt.figure(figsize=(10,5))
plt.hist(daily['iso_score'], bins=50)
plt.title("IsolationForest Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.close() 


import requests

def send_anomalies(payload, backend_url="http://localhost:5000/api/anomalies"):
    try:
        resp = requests.post(backend_url, json=payload, timeout=30)
        resp.raise_for_status()
        print("✅ Sent anomalies to backend:", resp.status_code)
    except Exception as e:
        print("❌ Failed to send anomalies:", e)

# Prepare payload
anomalies = daily[daily['iso_flag'] == True].copy()

anomalies['day'] = anomalies['day'].dt.strftime("%Y-%m-%d")
cols_to_send = ['custid','custname','day','total_cashin','txn_count','avg_txn_size','iso_score','zscore']
payload = anomalies[cols_to_send].to_dict(orient='records')

# Send to backend
send_anomalies(payload)




