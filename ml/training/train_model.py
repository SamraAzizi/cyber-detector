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




    def load_data(self):
        """Load processed training data"""
        try:
            X_train = pd.read_csv('ml/datasets/processed/X_train.csv')
            y_train = pd.read_csv('ml/datasets/processed/y_train.csv').squeeze()
            X_val = pd.read_csv('ml/datasets/processed/X_val.csv')
            y_val = pd.read_csv('ml/datasets/processed/y_val.csv').squeeze()
            
            logger.info(f"Training data loaded: {X_train.shape}, {y_train.shape}")
            logger.info(f"Validation data loaded: {X_val.shape}, {y_val.shape}")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            sys.exit(1)


            

            try:
                # Get model with default or configured params
                model = self._get_model_instance(model_name)
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = evaluator.evaluate(model, X_val, y_val)
                results[model_name] = metrics
                
                # Save model
                self._save_model(model, model_name, metrics)
                
                # Track best model
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    self.best_model = (model_name, model, metrics)
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                
                
        # Save comparison results
        with open(self.model_dir / 'model_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save best model separately
        if self.best_model:
            name, model, metrics = self.best_model
            logger.info(f"Best model: {name} with F1: {metrics['f1']:.4f}")
            self._save_model(model, f"best_{name}", metrics, best=True)
            
        return self.best_model
    



    def _get_model_instance(self, model_name):
        """Get configured model instance"""
        params = self.config.get('MODEL_PARAMS', {}).get(model_name, {})
        
        if model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                class_weight='balanced',
                random_state=self.config['RANDOM_STATE'],
                n_jobs=-1,
                **params
            )
        elif model_name == 'xgboost':
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                scale_pos_weight=self._calculate_scale_pos_weight(),
                random_state=self.config['RANDOM_STATE'],
                n_jobs=-1,
                **params
            )
        else:
            return self.models[model_name](**params)





    def train_models(self):
        """Train multiple models and select the best one"""
        X_train, y_train, X_val, y_val = self.load_data()
        evaluator = ModelEvaluator()
        best_score = -1
        
        results = {}
        
        for model_name, model_class in self.models.items():
            logger.info(f"Training {model_name}...")
            




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