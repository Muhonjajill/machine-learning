#!/usr/bin/env python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
import uvicorn
import os
import socket
import logging

from dotenv import load_dotenv
load_dotenv()

from models.anomaly_detector import AnomalyDetector
from models.forecast_model import ForecastModel
from scheduler import start_scheduler

# ─── Logging (console + rotating file) ───────────────────
"""Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            "logs/app.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3               # keep last 3 files
        )
    ]
)
logger = logging.getLogger(__name__)"""

Path("logs").mkdir(exist_ok=True)

# This config applies to ALL loggers across ALL modules
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False, 
    "formatters": {
        "detailed": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 5242880,  # 5MB
            "backupCount": 3,
            "encoding": "utf8"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
})

logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────
app = FastAPI(title="Transaction Monitoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared model instances ───────────────────────────────
# These are created ONCE and live in RAM for the lifetime
# of the server process. The scheduler receives references
# to these same objects, so training updates are immediately
# visible to API endpoints with no file I/O needed.
anomaly_detector = AnomalyDetector()
forecast_model = ForecastModel()


# ─── Pydantic schemas ─────────────────────────────────────
class Transaction(BaseModel):
    trandate: datetime
    terminalid: str
    cashin: float
    custid: str
    custname: str

class AnomalyResponse(BaseModel):
    custid: str
    custname: str
    day: str
    total_cashin: float
    txn_count: int
    avg_txn_size: float
    iso_score: float
    zscore: float
    is_anomaly: bool
    confidence: float


# ─── Startup ──────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Transaction Monitoring API...")
    try:
        # Pass the already-created model instances into the scheduler.
        # The scheduler will call .train() on these same objects,
        # so self._anomalies and self._forecasts get populated in-place.
        start_scheduler(anomaly_detector, forecast_model)
        logger.info("✅ Scheduler started — initial training running in background")
    except Exception as e:
        logger.error(f"❌ Scheduler failed to start: {e}")
        logger.info("Server will continue — trigger training manually via API")


# ─── Health ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "anomaly_model_ready": anomaly_detector.is_trained(),
        "forecast_model_ready": forecast_model.is_trained(),
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─── Real-time scoring ────────────────────────────────────
@app.post("/api/transactions/score", response_model=AnomalyResponse)
async def score_transaction(transaction: Transaction):
    """
    Score a single incoming deposit in real time.
    Uses the model already in RAM — no DB call needed here.
    """
    if transaction.cashin <= 0:
        raise HTTPException(status_code=400, detail="cashin must be > 0")
    if not anomaly_detector.is_trained():
        raise HTTPException(status_code=503, detail="Model not ready yet — training in progress")
    try:
        result = anomaly_detector.score_online(transaction.dict())
        return AnomalyResponse(**result)
    except Exception as e:
        logger.error(f"❌ Online scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Anomaly results ──────────────────────────────────────
@app.get("/api/anomalies")
async def get_anomalies():
    """Return anomalies detected in the last training run."""
    return anomaly_detector.get_all_anomalies()


# ─── Forecast results ─────────────────────────────────────
@app.get("/api/cashForecasts")
async def get_forecasts():
    """Return forecasts grouped by terminal."""
    forecasts = forecast_model.get_grouped_forecasts()
    if not forecasts:
        raise HTTPException(status_code=404, detail="No forecast data yet — training may still be running")
    return forecasts


# ─── Model status ─────────────────────────────────────────
@app.get("/api/models/status")
async def model_status():
    return {
        "anomaly_detector": {
            "trained": anomaly_detector.is_trained(),
            "last_trained": anomaly_detector.last_trained,
            "confidence": anomaly_detector.global_confidence,
            "anomalies_in_memory": len(anomaly_detector.get_all_anomalies()),
        },
        "forecast_model": {
            "trained": forecast_model.is_trained(),
            "last_trained": forecast_model.last_trained,
            "records_in_memory": len(forecast_model._forecasts),
        }
    }


# ─── Manual training triggers ─────────────────────────────
@app.post("/api/train/anomaly")
async def trigger_anomaly(background_tasks: BackgroundTasks):
    """Manually kick off anomaly training outside the schedule."""
    background_tasks.add_task(anomaly_detector.train)
    return {"status": "Anomaly training started in background"}

@app.post("/api/train/forecast")
async def trigger_forecast(background_tasks: BackgroundTasks):
    """Manually kick off forecast training outside the schedule."""
    background_tasks.add_task(forecast_model.train)
    return {"status": "Forecast training started in background"}


# ─── Entry point ──────────────────────────────────────────
def find_available_port(start=8000, attempts=10):
    for port in range(start, start + attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            logger.warning(f"Port {port} in use, trying next...")
    raise RuntimeError("No available ports found")


if __name__ == "__main__":
    HOST = os.getenv("API_HOST", "127.0.0.1")
    PORT = int(os.getenv("API_PORT", 8000))
    RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

    try:
        uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD, log_level="info")
    except OSError:
        port = find_available_port(PORT)
        logger.info(f"Retrying on port {port}")
        uvicorn.run("main:app", host=HOST, port=port, reload=RELOAD, log_level="info")