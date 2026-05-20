#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pyodbc
import pickle
import os
import logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'total_cashin', 'txn_count', 'avg_txn_size',
            'rolling_mean_7', 'rolling_std_7',
            'pct_change', 'volatility_7'
        ]
        self.global_confidence = 0.0
        self.last_trained = None
        self._anomalies = []            # results in RAM
        self.historical_data = pd.DataFrame()  # used for online scoring only

        # Only persist the trained model files
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.model_path = self.data_dir / "anomaly_model.pkl"
        self.scaler_path = self.data_dir / "anomaly_scaler.pkl"

        self._load_model()

    # ─── DB ───────────────────────────────────────────────
    def _get_db_connection(self):
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('DB_SERVER')};"
            f"DATABASE={os.getenv('DB_DATABASE')};"
            f"UID={os.getenv('DB_USERNAME')};"
            f"PWD={os.getenv('DB_PASSWORD')};"
            f"TrustServerCertificate=yes;"
        )
        return pyodbc.connect(connection_string)

    # ─── Model persistence ────────────────────────────────
    def _load_model(self):
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Loaded existing anomaly detection model from disk")
        except Exception as e:
            logger.warning(f"⚠️ Could not load model: {e}")

    def _save_model(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("💾 Model and scaler saved to disk")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

    # ─── Status ───────────────────────────────────────────
    def is_trained(self):
        return self.model is not None and self.scaler is not None

    # ─── Results (in-memory only) ─────────────────────────
    def store_anomalies(self, anomalies):
        now = datetime.now().isoformat()
        self._anomalies = [{'received_at': now, **a} for a in anomalies]
        logger.info(f"📦 Stored {len(self._anomalies)} anomalies in memory")
        return len(self._anomalies)

    def get_all_anomalies(self):
        return self._anomalies

    # ─── Online scoring ───────────────────────────────────
    def score_online(self, transaction):
        """
        Real-time scoring for a single incoming transaction.
        Uses self.historical_data (last 180 days in RAM) to
        calculate rolling features for the customer, then
        scores against the trained IsolationForest model.
        """
        if not self.is_trained():
            raise Exception("Model not trained. Wait for initial training to complete.")

        tx = pd.DataFrame([transaction])
        tx['trandate'] = pd.to_datetime(tx['trandate'])
        tx['day'] = tx['trandate'].dt.date

        # Append new transaction to in-memory history
        if len(self.historical_data) > 0:
            self.historical_data = pd.concat(
                [self.historical_data, tx], ignore_index=True
            )
        else:
            self.historical_data = tx.copy()

        # Keep only last 180 days to prevent unbounded RAM growth
        cutoff = datetime.now() - timedelta(days=180)
        self.historical_data = self.historical_data[
            pd.to_datetime(self.historical_data['trandate']) >= cutoff
        ]

        daily = self._calculate_daily_features(tx['custid'].iloc[0])

        # Not enough history yet for this customer — return safe default
        if len(daily) == 0:
            return {
                'custid': tx['custid'].iloc[0],
                'custname': tx['custname'].iloc[0],
                'day': str(tx['day'].iloc[0]),
                'total_cashin': float(tx['cashin'].iloc[0]),
                'txn_count': 1,
                'avg_txn_size': float(tx['cashin'].iloc[0]),
                'iso_score': 0.0,
                'zscore': 0.0,
                'is_anomaly': False,
                'confidence': 0.5
            }

        latest = daily.iloc[-1]
        X = latest[self.features].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        iso_score = self.model.decision_function(X_scaled)[0]
        is_anomaly = self.model.predict(X_scaled)[0] == -1

        return {
            'custid': latest['custid'],
            'custname': latest['custname'],
            'day': str(latest['day']),
            'total_cashin': float(latest['total_cashin']),
            'txn_count': int(latest['txn_count']),
            'avg_txn_size': float(latest['avg_txn_size']),
            'iso_score': float(iso_score),
            'zscore': float(latest.get('zscore', 0)),
            'is_anomaly': bool(is_anomaly),
            'confidence': float(self.global_confidence) if self.global_confidence > 0 else 0.5
        }

    def _calculate_daily_features(self, custid):
        customer_data = self.historical_data[
            self.historical_data['custid'] == custid
        ].copy()

        if len(customer_data) == 0:
            return pd.DataFrame()

        customer_data['day'] = pd.to_datetime(customer_data['trandate']).dt.date

        daily = customer_data.groupby(['custid', 'custname', 'day']).agg(
            total_cashin=('cashin', 'sum'),
            txn_count=('cashin', 'count'),
            avg_txn_size=('cashin', 'mean'),
            max_txn=('cashin', 'max'),
            min_txn=('cashin', 'min')
        ).reset_index()

        daily['day'] = pd.to_datetime(daily['day'])
        daily = daily.sort_values('day')

        daily['rolling_mean_7'] = daily['total_cashin'].rolling(7, min_periods=1).mean()
        daily['rolling_std_7'] = daily['total_cashin'].rolling(7, min_periods=1).std()
        daily['rolling_txn_count_7'] = daily['txn_count'].rolling(7, min_periods=1).mean()
        daily['pct_change'] = daily['total_cashin'].pct_change()
        daily['volatility_7'] = daily['total_cashin'].rolling(7, min_periods=1).std()
        daily['zscore'] = (
            (daily['total_cashin'] - daily['rolling_mean_7']) / daily['rolling_std_7']
        )

        return daily.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ─── Batch training (called by scheduler) ─────────────
    def train(self):
        logger.info("🔄 Starting anomaly detection training...")
        try:
            conn = self._get_db_connection()
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            sql = f"""
                SELECT trandate, terminalid, cashin, custid, custname
                FROM transactions
                WHERE trandate >= '{start_date}' AND cashin > 0
                ORDER BY terminalid, trandate
            """

            df = pd.read_sql(sql, conn)
            conn.close()

            if len(df) == 0:
                logger.warning("⚠️ No data to train on")
                return

            # Refresh historical_data from DB — this replaces the
            # incrementally-grown online scoring history with clean data
            self.historical_data = df.copy()

            df['trandate'] = pd.to_datetime(df['trandate'])
            df['day'] = df['trandate'].dt.date

            daily = df.groupby(['custid', 'custname', 'day']).agg(
                total_cashin=('cashin', 'sum'),
                txn_count=('cashin', 'count'),
                avg_txn_size=('cashin', 'mean'),
                max_txn=('cashin', 'max'),
                min_txn=('cashin', 'min')
            ).reset_index()

            daily['day'] = pd.to_datetime(daily['day'])
            daily = daily.sort_values(['custid', 'day'])

            daily['rolling_mean_7'] = daily.groupby('custid')['total_cashin'] \
                .transform(lambda s: s.rolling(7, min_periods=3).mean())
            daily['rolling_std_7'] = daily.groupby('custid')['total_cashin'] \
                .transform(lambda s: s.rolling(7, min_periods=3).std())
            daily['rolling_txn_count_7'] = daily.groupby('custid')['txn_count'] \
                .transform(lambda s: s.rolling(7, min_periods=3).mean())
            daily['pct_change'] = daily.groupby('custid')['total_cashin'].pct_change()
            daily['volatility_7'] = daily.groupby('custid')['total_cashin'] \
                .transform(lambda s: s.rolling(7, min_periods=3).std())
            daily['zscore'] = (
                (daily['total_cashin'] - daily['rolling_mean_7']) / daily['rolling_std_7']
            )
            daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)

            X = daily[self.features].fillna(0).values
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = IsolationForest(
                n_estimators=200, contamination=0.01, random_state=42
            )
            self.model.fit(X_scaled)

            self.global_confidence = self._calculate_confidence(X_scaled, daily)
            self._save_model()
            self.last_trained = datetime.now().isoformat()

            daily['iso_flag'] = self.model.predict(X_scaled) == -1
            daily['iso_score'] = self.model.decision_function(X_scaled)

            anomalies = daily[daily['iso_flag']].copy()
            anomalies['day'] = anomalies['day'].dt.strftime("%Y-%m-%d")

            payload = anomalies[
                ['custid', 'custname', 'day', 'total_cashin',
                 'txn_count', 'avg_txn_size', 'iso_score', 'zscore']
            ].to_dict(orient='records')

            self.store_anomalies(payload)

            logger.info(f"✅ Training complete — {len(anomalies)} anomalies detected")
            logger.info(f"📊 Confidence: {self.global_confidence:.3f}")

        except Exception as e:
            logger.error(f"❌ Anomaly training failed: {e}")
            raise

    def _calculate_confidence(self, X_scaled, daily):
        try:
            stability = self._stability_test(X_scaled)
            scores = self.model.decision_function(X_scaled)
            preds = self.model.predict(X_scaled)
            separation = np.mean(scores[preds == 1]) - np.mean(scores[preds == -1])
            sep_norm = np.tanh(separation)
            anomaly_rate = np.mean(preds == -1)
            confidence = 0.5 * stability + 0.3 * sep_norm + 0.2 * (1 - anomaly_rate)
            return float(np.clip(confidence, 0, 1))
        except Exception:
            return 0.5

    def _stability_test(self, X, runs=5, sample_fraction=0.8):
        try:
            flags = []
            for i in range(runs):
                iso_tmp = IsolationForest(
                    n_estimators=200, contamination=0.01, random_state=42 + i
                )
                idx = np.random.choice(len(X), int(len(X) * sample_fraction), replace=False)
                iso_tmp.fit(X[idx])
                flags.append((iso_tmp.predict(X) == -1).astype(int))
            flags = np.array(flags)
            similarities = [
                np.mean(flags[i] == flags[j])
                for i in range(runs) for j in range(i + 1, runs)
            ]
            return float(np.mean(similarities))
        except Exception:
            return 0.5