"""
Cyber Threat Detection Model Training Pipeline

Features:
- Multi-model training (RF, XGBoost, SVM, MLP)
- MLflow experiment tracking
- Automated drift detection baselines
- Production-ready logging and error handling
"""

# Standard Library
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Third-Party
import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Local
from ..utils.config import load_config
from .evaluate_model import ModelEvaluator

# 1. Logging Configuration =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml/training/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)



# 2. Main Training Class =======================================================
class ModelTrainer:
    """Orchestrates model training, evaluation, and registration."""

    def __init__(self, retrain: bool = False, retrain_reason: str = "scheduled"):
        """Initialize trainer with configuration and tracking.
        
        Args:
            retrain: Whether this is a retraining job
            retrain_reason: Reason for retraining (drift/scheduled/performance)
        """
        self.config = load_config()
        self.models = {
            "random_forest": RandomForestClassifier,
            "xgboost": XGBClassifier,
            "svm": SVC,
            "mlp": MLPClassifier,
        }
        self.best_model = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = Path(self.config["MODEL_SAVE_DIR"]) / self.timestamp
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.retrain = retrain
        self.retrain_reason = retrain_reason


        