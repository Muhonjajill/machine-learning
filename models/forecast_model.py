#!/usr/bin/env python
import pandas as pd
import numpy as np
import pyodbc
import os
import logging
import warnings
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

load_dotenv()
logger = logging.getLogger(__name__)


class ForecastModel:
    def __init__(self):
        self.last_trained = None
        self._forecasts = []    # results in RAM, no files

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

    # ─── Status ───────────────────────────────────────────
    def is_trained(self):
        return len(self._forecasts) > 0

    # ─── Results (in-memory only) ─────────────────────────
    def store_forecasts(self, forecasts):
        try:
            normalized = []
            for f in forecasts:
                normalized.append({
                    'terminalid': f.get('terminalid'),
                    'terminal_name': f.get('terminal_name', ''),
                    'location': f.get('location', ''),
                    'date': f.get('ds'),
                    'yhat': float(f.get('yhat', 0)),
                    'yhat_lower': float(f.get('yhat_lower', 0)),
                    'yhat_upper': float(f.get('yhat_upper', 0)),
                    'actual': float(f.get('actual', 0) or f.get('actual_cashin', 0) or 0),
                    'difference': float(f.get('difference', 0)),
                    'percent_diff': float(f.get('percent_diff', 0)),
                    'anomaly': f.get('anomaly', 'NORMAL'),
                    'z_anomaly': f.get('z_anomaly', 'NORMAL'),
                    'final_anomaly': f.get('final_anomaly', 'NORMAL'),
                    'zscore': float(f.get('zscore', 0) or 0),
                    'roll_mean': float(f.get('roll_mean', 0) or 0),
                    'roll_std': float(f.get('roll_std', 0) or 0),
                })
            self._forecasts = normalized
            logger.info(f"📦 Stored {len(self._forecasts)} forecast records in memory")
            return len(self._forecasts)
        except Exception as e:
            logger.error(f"❌ Failed to store forecasts: {e}")
            return 0

    def get_grouped_forecasts(self):
        if not self._forecasts:
            return []

        grouped = {}
        for row in self._forecasts:
            tid = row['terminalid']
            if tid not in grouped:
                grouped[tid] = []
            grouped[tid].append(row)

        result = []
        for terminal_id, rows in grouped.items():
            rows.sort(key=lambda x: x['date'])
            y_values = [r['yhat'] for r in rows]
            trend = (
                "increasing" if y_values[-1] > y_values[0] * 1.05 else
                "decreasing" if y_values[-1] < y_values[0] * 0.95 else
                "stable"
            )
            avg = sum(y_values) / len(y_values) if y_values else 0
            result.append({
                'terminalId': terminal_id,
                'trend': trend,
                'rows': rows,
                'summary': {
                    'avgForecast': round(avg),
                    'anomalies': {
                        'high': len([x for x in rows if x['anomaly'] == 'HIGH']),
                        'low': len([x for x in rows if x['anomaly'] == 'LOW']),
                        'normal': len([x for x in rows if x['anomaly'] == 'NORMAL']),
                    }
                }
            })
        return result

    # ─── Helpers ──────────────────────────────────────────
    def _simple_forecast(self, data, periods=10):
        ma_7 = data['y'].rolling(window=7, min_periods=1).mean()
        trend = data['y'].pct_change().mean()
        last_ma = ma_7.iloc[-1]
        std = data['y'].std()
        last_date = pd.to_datetime(data['ds'].iloc[-1])
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=periods
        )
        forecasts = []
        for i, d in enumerate(future_dates):
            val = max(0, last_ma * (1 + trend * (i + 1)))
            forecasts.append({
                'ds': d,
                'yhat': val,
                'yhat_lower': max(0, val - 1.5 * std),
                'yhat_upper': val + 1.5 * std,
            })
        historical = data.copy()
        historical['yhat'] = ma_7
        historical['yhat_lower'] = (historical['y'] - 1.5 * std).clip(lower=0)
        historical['yhat_upper'] = historical['y'] + 1.5 * std
        return pd.concat(
            [historical[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
             pd.DataFrame(forecasts)],
            ignore_index=True
        )

    def _zscore_flag(self, z):
        if pd.isna(z): return "NORMAL"
        if z > 2.5: return "SPIKE"
        if z < -2.5: return "DROP"
        return "NORMAL"

    def _final_flag(self, row):
        p, z = row['anomaly'], row['z_anomaly']
        if p == "HIGH" and z == "SPIKE": return "HIGH SPIKE"
        if p == "LOW" and z == "DROP": return "LOW DROP"
        if p == "HIGH" and row.get('zscore', 0) > 2: return "HIGH (Z)"
        if p == "LOW" and row.get('zscore', 0) < -2: return "LOW (Z)"
        return "NORMAL"

    # ─── Batch training (called by scheduler) ─────────────
    def train(self):
        logger.info("🔄 Starting forecast training...")
        try:
            conn = self._get_db_connection()

            query = """
                SELECT t.terminalid, t.name AS terminal_name, t.location,
                       tr.trandate, tr.cashin
                FROM [transactions] AS tr
                JOIN [terminals] AS t ON tr.terminalid = t.terminalid
                WHERE tr.trandate >= '2025-01-01' AND tr.cashin IS NOT NULL
            """
            df = pd.read_sql(query, conn)
            df['trandate'] = pd.to_datetime(df['trandate'])
            df['cashin'] = pd.to_numeric(df['cashin'], errors='coerce')

            df = df.groupby(
                ['terminalid', 'terminal_name', 'location',
                 pd.Grouper(key='trandate', freq='D')],
                dropna=False
            ).agg({'cashin': 'sum'}).reset_index()

            terminals = df['terminalid'].unique()
            logger.info(f"📊 Forecasting for {len(terminals)} terminals")

            all_forecasts = []
            use_prophet = True

            for terminal in terminals:
                terminal_data = df[
                    df['terminalid'] == terminal
                ].dropna(subset=['cashin'])

                if len(terminal_data) < 3:
                    logger.warning(f"⚠️ Skipping {terminal} — insufficient data")
                    continue

                data = (
                    terminal_data.groupby('trandate')['cashin'].sum()
                    .reset_index()
                    .rename(columns={'trandate': 'ds', 'cashin': 'y'})
                )
                data = data.drop_duplicates(subset=['ds'], keep='last').sort_values('ds')

                try:
                    if use_prophet:
                        from prophet import Prophet
                        import sys
                        from io import StringIO

                        model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            yearly_seasonality=False,
                            changepoint_prior_scale=0.1,
                            seasonality_mode='additive',
                            interval_width=0.80,
                        )
                        # Suppress Prophet's verbose stdout
                        old_out, old_err = sys.stdout, sys.stderr
                        sys.stdout = sys.stderr = StringIO()
                        try:
                            model.fit(data)
                            future = model.make_future_dataframe(periods=10)
                            forecast = model.predict(future)
                        finally:
                            sys.stdout, sys.stderr = old_out, old_err

                        actuals = data['y'].values
                        preds = model.predict(data)['yhat'].values
                        std_error = np.std(actuals - preds)

                        forecast['yhat'] = forecast['yhat'].clip(lower=0)
                        forecast['yhat_lower'] = (
                            forecast['yhat'] - 1.5 * std_error
                        ).clip(lower=0)
                        forecast['yhat_upper'] = (
                            forecast['yhat'] + 1.5 * std_error
                        ).clip(upper=data['y'].mean() * 3)

                except Exception as prophet_err:
                    logger.warning(
                        f"⚠️ Prophet failed for {terminal}, "
                        f"falling back to simple forecast: {prophet_err}"
                    )
                    use_prophet = False
                    forecast = self._simple_forecast(data, periods=10)

                forecast['terminalid'] = terminal
                info = terminal_data[['terminal_name', 'location']].iloc[0]
                forecast['terminal_name'] = info['terminal_name']
                forecast['location'] = info['location']
                all_forecasts.append(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper',
                               'terminalid', 'terminal_name', 'location']]
                )

            if not all_forecasts:
                logger.warning("⚠️ No forecasts generated")
                return

            all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
            start_date = all_forecasts_df['ds'].min().strftime('%Y-%m-%d')

            actual_query = """
                SELECT terminalid,
                       CAST(trandate AS DATE) AS trandate,
                       SUM(cashin) AS actual_cashin
                FROM transactions
                WHERE trandate >= ?
                GROUP BY terminalid, CAST(trandate AS DATE)
            """
            actual_df = pd.read_sql(actual_query, conn, params=[start_date])
            conn.close()

            all_forecasts_df['ds'] = pd.to_datetime(all_forecasts_df['ds']).dt.date
            actual_df['trandate'] = pd.to_datetime(actual_df['trandate']).dt.date

            merged = pd.merge(
                all_forecasts_df, actual_df,
                left_on=['terminalid', 'ds'],
                right_on=['terminalid', 'trandate'],
                how='left'
            )

            merged['actual'] = merged['actual_cashin']
            merged['difference'] = merged['actual_cashin'] - merged['yhat']
            merged['percent_diff'] = (merged['difference'] / merged['yhat']) * 100
            merged = merged.sort_values(by=['terminalid', 'ds'])

            merged['roll_mean'] = merged.groupby('terminalid')['actual_cashin'] \
                .transform(lambda x: x.rolling(window=14, min_periods=3).mean())
            merged['roll_std'] = merged.groupby('terminalid')['actual_cashin'] \
                .transform(lambda x: x.rolling(window=14, min_periods=3).std())
            merged['roll_std'] = merged['roll_std'].replace(0, np.nan)
            merged['zscore'] = (
                (merged['actual_cashin'] - merged['roll_mean']) / merged['roll_std']
            )

            merged['z_anomaly'] = merged['zscore'].apply(self._zscore_flag)
            merged['anomaly'] = merged.apply(
                lambda r: "HIGH" if r.actual_cashin > r.yhat_upper else
                          "LOW" if r.actual_cashin < r.yhat_lower else "NORMAL",
                axis=1
            )
            merged['final_anomaly'] = merged.apply(self._final_flag, axis=1)

            # Clean for serialization
            for col in ['ds', 'trandate']:
                if col in merged.columns:
                    merged[col] = merged[col].astype(str)
            for col in ['difference', 'percent_diff', 'actual_cashin', 'actual',
                        'yhat', 'yhat_lower', 'yhat_upper',
                        'roll_mean', 'roll_std', 'zscore']:
                if col in merged.columns:
                    merged[col] = (
                        pd.to_numeric(merged[col], errors='coerce')
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0)
                    )
            merged = merged.where(pd.notnull(merged), None)

            self.store_forecasts(merged.to_dict(orient='records'))
            self.last_trained = datetime.now().isoformat()

            method = "Prophet" if use_prophet else "Simple Moving Average"
            logger.info(f"✅ Forecast training complete using {method}")
            logger.info(f"📊 Generated {len(merged)} forecast records")

        except Exception as e:
            logger.error(f"❌ Forecast training failed: {e}")
            import traceback
            traceback.print_exc()
            raise