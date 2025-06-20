import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from ..utils.config import load_config
from .evaluate_model import ModelEvaluator
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml/training/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.config = load_config()
        self.models = {
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'svm': SVC,
            'mlp': MLPClassifier
        }
        self.best_model = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = Path(self.config['MODEL_SAVE_DIR']) / self.timestamp
        self.model_dir.mkdir(parents=True, exist_ok=True)


    def _calculate_scale_pos_weight(self):
        """Calculate class weight for XGBoost"""
        y_train = pd.read_csv('ml/datasets/processed/y_train.csv').squeeze()
        class_counts = y_train.value_counts()
        return class_counts[0] / class_counts[1]

    def _save_model(self, model, name, metrics, best=False):
        """Save model and its metrics"""
        prefix = 'best_' if best else ''
        model_path = self.model_dir / f'{prefix}{name}.joblib'
        
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} model to {model_path}")
        
        # Save metrics
        metrics_path = self.model_dir / f'{prefix}{name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def run(self):
        """Execute full training pipeline"""
        try:
            logger.info("Starting model training pipeline")
            best_model = self.train_models()
            
            if best_model:
                name, model, metrics = best_model
                logger.info(f"Training completed. Best model: {name}")
                logger.info(f"Validation F1: {metrics['f1']:.4f}")
                return True
            else:
                logger.error("No models trained successfully")
                return False
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return False

if __name__ == "__main__":
    trainer = ModelTrainer()
    success = trainer.run()
    sys.exit(0 if success else 1)