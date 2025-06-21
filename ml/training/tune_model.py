import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..utils.config import load_config
from .evaluate_model import ModelEvaluator
import joblib
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelTuner:
    def __init__(self, model_type='random_forest'):
        self.config = load_config()
        self.model_type = model_type
        self.evaluator = ModelEvaluator()
        self.study = None
        
    def load_data(self):
        """Load processed training data"""
        X_train = pd.read_csv('ml/datasets/processed/X_train.csv')
        y_train = pd.read_csv('ml/datasets/processed/y_train.csv').squeeze()
        return X_train, y_train
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        X_train, y_train = self.load_data()

        