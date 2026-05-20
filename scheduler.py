#!/usr/bin/env python
"""
Background scheduler for daily model training
Runs anomaly detection and forecasting at scheduled times
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_scheduler(anomaly_detector, forecast_model):
    """
    Initialize and start the background scheduler
    
    Args:
        anomaly_detector: Instance of AnomalyDetector
        forecast_model: Instance of ForecastModel
    """
    scheduler = BackgroundScheduler()
    
    # Schedule anomaly detection training daily at 2 AM
    scheduler.add_job(
        func=train_anomaly_detection,
        trigger=CronTrigger(hour=14, minute=0),
        args=[anomaly_detector],
        id='anomaly_training',
        name='Daily Anomaly Detection Training',
        replace_existing=True
    )
    
    # Schedule forecast training daily at 3 AM
    scheduler.add_job(
        func=train_forecast,
        trigger=CronTrigger(hour=15, minute=0),
        args=[forecast_model],
        id='forecast_training',
        name='Daily Forecast Training',
        replace_existing=True
    )
    
    # Optional: Run initial training on startup (comment out if not needed)
    scheduler.add_job(
        func=initial_training,
        trigger='date',
        args=[anomaly_detector, forecast_model],
        id='initial_training',
        name='Initial Model Training'
    )
    
    scheduler.start()
    logger.info("✅ Scheduler started successfully")
    logger.info("📅 Anomaly training scheduled: Daily at 2:00 PM")
    logger.info("📅 Forecast training scheduled: Daily at 3:00 PM")

def initial_training(anomaly_detector, forecast_model):
    """Run initial training when server starts"""
    logger.info("🚀 Running initial model training...")
    try:
        train_anomaly_detection(anomaly_detector)
        train_forecast(forecast_model)
        logger.info("✅ Initial training complete")
    except Exception as e:
        logger.error(f"❌ Initial training failed: {e}")

def train_anomaly_detection(anomaly_detector):
    """Scheduled job: Train anomaly detection model"""
    logger.info("🔄 Starting scheduled anomaly detection training...")
    try:
        anomaly_detector.train()
        logger.info("✅ Anomaly detection training completed")
    except Exception as e:
        logger.error(f"❌ Anomaly detection training failed: {e}")

def train_forecast(forecast_model):
    """Scheduled job: Train forecast model"""
    logger.info("🔄 Starting scheduled forecast training...")
    try:
        forecast_model.train()
        logger.info("✅ Forecast training completed")
    except Exception as e:
        logger.error(f"❌ Forecast training failed: {e}")