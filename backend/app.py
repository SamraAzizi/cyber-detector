from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from model_loader import ModelLoader
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Literal
import time
from datetime import datetime
import uuid
import logging
import os
from pathlib import Path
import yaml
import json
from cachetools import TTLCache
import platform
import psutil
import pandas as pd
from scipy.stats import ks_2samp
from apscheduler.schedulers.background import BackgroundScheduler
import mlflow
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppConfig:
    """Enhanced configuration management with monitoring defaults"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_config()
        return cls._instance
        


    def load_config(self):
        """Load and validate configuration with monitoring defaults"""
        try:
            with open("config/config.yaml") as f:
                self.config = yaml.safe_load(f) or {}
            self._set_monitoring_defaults()
            self._validate()
        except Exception as e:
            logger.critical(f"Config load failed: {str(e)}")
            raise
            


    def _set_monitoring_defaults(self):
        """Set monitoring-specific defaults"""
        monitoring_defaults = {
            'monitoring': {
                'drift_threshold': 0.05,
                'performance_window': 1000,
                'performance_decay_threshold': 0.05,
                'enable_scheduled_checks': True,
                'check_interval_hours': 24
            },
            'mlflow': {
                'tracking_uri': "file:./mlruns",
                'experiment_name': "cybershield_production"
            }
        }
        for section, values in monitoring_defaults.items():
            self.config.setdefault(section, values)
            


    def _validate(self):
        """Validate monitoring configuration"""
        if not Path(config['monitoring']['reference_stats_path']).exists():
            logger.warning("Reference stats not found - drift detection disabled")

try:
    app_config = AppConfig()
    config = app_config.config
except Exception as e:
    logger.critical(f"Configuration error: {str(e)}")
    raise




class DriftDetector:
    """Real-time feature drift detection"""

    def __init__(self):
        self.reference_stats = self._load_reference_stats()
        self.drift_log = Path("logs/drift_metrics.csv")
        


    def _load_reference_stats(self):
        """Load training data statistics"""
        try:
            with open(config['monitoring']['reference_stats_path']) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load reference stats: {str(e)}")
            return None
            


    def check_drift(self, features: dict):
        """Check feature drift using KS-test"""
        if not self.reference_stats:
            return None
            
        drift_results = {}
        for feat_name, value in features.items():
            if feat_name in self.reference_stats['features']:
                stat, pval = ks_2samp(
                    [value],
                    self.reference_stats['features'][feat_name]['samples']
                )
                drift_results[feat_name] = {
                    'ks_statistic': stat,
                    'p_value': pval,
                    'drift_detected': pval < config['monitoring']['drift_threshold']
                }
        
        self._log_drift(drift_results)
        return drift_results
        


    def _log_drift(self, results: dict):
        """Log drift metrics to persistent storage"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_features': [k for k,v in results.items() if v['drift_detected']],
            'max_ks_statistic': max(v['ks_statistic'] for v in results.values())
        }
        
        self.drift_log.parent.mkdir(exist_ok=True)
        pd.DataFrame([log_entry]).to_csv(
            self.drift_log,
            mode='a',
            header=not self.drift_log.exists(),
            index=False
        )



class PerformanceTracker:
    """Model performance monitoring"""
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.performance_log = Path("logs/performance_metrics.csv")
        


    def add_prediction(self, y_pred: float, y_true: float = None):
        """Record prediction with optional ground truth"""
        self.predictions.append(y_pred)
        if y_true is not None:
            self.actuals.append(y_true)
            self._check_performance()
            


    def _check_performance(self):
        """Check for performance decay"""
        if len(self.actuals) < config['monitoring']['performance_window']:
            return
            
        window = config['monitoring']['performance_window']
        current_acc = accuracy_score(self.actuals[-window:], [round(p) for p in self.predictions[-window:]])
        baseline_acc = self._get_baseline_accuracy()
        
        decay = baseline_acc - current_acc
        significant_decay = decay > config['monitoring']['performance_decay_threshold']
        
        self._log_performance(current_acc, baseline_acc, significant_decay)
        return significant_decay
        


    def _get_baseline_accuracy(self):
        """Get original validation accuracy"""
        with open('ml/models/deployment_packages/latest/metrics.json') as f:
            return json.load(f)['accuracy']
            


    def _log_performance(self, current_acc: float, baseline_acc: float, decay_detected: bool):
        """Log performance metrics"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'current_accuracy': current_acc,
            'baseline_accuracy': baseline_acc,
            'decay_detected': decay_detected
        }
        
        self.performance_log.parent.mkdir(exist_ok=True)
        pd.DataFrame([log_entry]).to_csv(
            self.performance_log,
            mode='a',
            header=not self.performance_log.exists(),
            index=False
        )




class ModelRegistry:
    """Version control and model management"""
    def __init__(self):
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def log_retraining(self, model, metrics: dict, reason: str):
        """Log a new model version"""
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_params({
                'retrain_reason': reason,
                'model_type': type(model).__name__
            })
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(config['monitoring']['reference_stats_path'])
            
        logger.info(f"Logged model version: {mlflow.active_run().info.run_id}")


app = FastAPI(
    title="CyberShield API",
    description="Advanced Threat Detection with Monitoring",
    version=config['version']
)

# Initialize components
drift_detector = DriftDetector()
performance_tracker = PerformanceTracker()
model_registry = ModelRegistry()
scheduler = BackgroundScheduler()



class MonitoringStats(BaseModel):
    drift_metrics: dict
    performance_metrics: dict
    model_versions: list

@app.get("/monitoring/stats", response_model=MonitoringStats)
async def get_monitoring_stats():
    """Get comprehensive monitoring metrics"""
    return {
        "drift_metrics": pd.read_csv("logs/drift_metrics.csv").tail(1).to_dict('records')[0] if Path("logs/drift_metrics.csv").exists() else {},
        "performance_metrics": pd.read_csv("logs/performance_metrics.csv").tail(1).to_dict('records')[0] if Path("logs/performance_metrics.csv").exists() else {},
        "model_versions": [run.info.run_id for run in mlflow.search_runs()]
    }


@app.post("/trigger-retraining", dependencies=[Depends(get_api_key)])
async def trigger_retraining(reason: str):
    """Manually trigger model retraining"""
    try:
        # In production, you'd call your training pipeline here
        # For example: subprocess.run(["python", "ml/training/train_model.py", "--retrain"])
        
        # Mock response for illustration
        return {
            "status": "success",
            "message": f"Retraining triggered: {reason}",
            "tracking_url": f"{config['mlflow']['tracking_uri']}/#/experiments/1/runs/1"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def check_model_health():
    """Scheduled monitoring job"""
    try:
        # 1. Check feature drift
        recent_features = pd.read_csv("logs/predictions.csv").tail(100)
        drift_report = drift_detector.check_drift(recent_features)
        
        # 2. Check performance decay
        performance_report = performance_tracker._check_performance()
        
        # 3. Trigger retraining if needed
        if (drift_report and sum(d['drift_detected'] for d in drift_report.values()) > 3) or performance_report:
            logger.warning("Conditions met for retraining!")
            # In production: trigger_retraining("automated: drift detected")
            
    except Exception as e:
        logger.error(f"Monitoring job failed: {str(e)}")

if config['monitoring']['enable_scheduled_checks']:
    scheduler.add_job(
        check_model_health,
        'interval',
        hours=config['monitoring']['check_interval_hours']
    )


@app.post("/detect-threat", response_model=DetectionResult)
async def detect_threat(data: NetworkData):
    # ... existing prediction code ...
    
    # Monitoring integration
    features_dict = dict(zip(model_adapter.metadata['input_features'], data.features))
    
    # 1. Check feature drift
    drift_results = drift_detector.check_drift(features_dict)
    
    # 2. Track performance (if ground truth available)
    if data.get('ground_truth'):
        performance_tracker.add_prediction(
            result['prediction'],
            data['ground_truth']
        )
    
    # 3. Add warnings if issues detected
    if drift_results and any(d['drift_detected'] for d in drift_results.values()):
        result['warnings'].append("Feature drift detected")
    
    return result


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring services"""
    scheduler.start()
    logger.info("Monitoring scheduler started")
    
    # Load initial model
    if not model_adapter.load_deployment(DEPLOYMENT_PACKAGE):
        logger.error("Model loading failed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup monitoring services"""
    scheduler.shutdown()
    logger.info("Monitoring scheduler stopped")
