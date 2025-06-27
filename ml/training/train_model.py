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
import argparse
from scipy.stats import ks_2samp
import mlflow

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
    def __init__(self, retrain=False, retrain_reason=""):
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
        self.retrain = retrain
        self.retrain_reason = retrain_reason

    def load_data(self):
        """Load processed training data with validation"""
        try:
            X_train = pd.read_csv('ml/datasets/processed/X_train.csv')
            y_train = pd.read_csv('ml/datasets/processed/y_train.csv').squeeze()
            X_val = pd.read_csv('ml/datasets/processed/X_val.csv')
            y_val = pd.read_csv('ml/datasets/processed/y_val.csv').squeeze()
            
            logger.info(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}")
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            sys.exit(1)

    def train_models(self):
        """Train and evaluate multiple models"""
        X_train, y_train, X_val, y_val = self.load_data()
        evaluator = ModelEvaluator()
        best_score = -1
        results = {}
        
        # Generate reference stats for drift detection
        if self.retrain:
            self._generate_reference_stats(X_train)

        for model_name, model_class in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                model = self._get_model_instance(model_name)
                model.fit(X_train, y_train)
                
                metrics = evaluator.evaluate(model, X_val, y_val)
                results[model_name] = metrics
                
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    self.best_model = (model_name, model, metrics)

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")

        self._save_results(results)
        return self.best_model

    def _get_model_instance(self, model_name):
        """Configure model with parameters"""
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

    def _calculate_scale_pos_weight(self):
        """Handle class imbalance for XGBoost"""
        y_train = pd.read_csv('ml/datasets/processed/y_train.csv').squeeze()
        class_counts = y_train.value_counts()
        return class_counts[0] / class_counts[1]

    def _generate_reference_stats(self, X_train):
        """Create baseline stats for drift detection"""
        stats = {
            'features': {},
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'retrain_reason': self.retrain_reason
            }
        }
        
        for col in X_train.columns:
            samples = X_train[col].sample(min(1000, len(X_train))).values
            stats['features'][col] = {
                'mean': float(X_train[col].mean()),
                'std': float(X_train[col].std()),
                'samples': samples.tolist()
            }
        
        stats_path = self.model_dir / 'reference_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved reference stats to {stats_path}")

    def _save_results(self, results):
        """Save all training artifacts"""
        # Save model comparison
        with open(self.model_dir / 'model_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best model
        if self.best_model:
            name, model, metrics = self.best_model
            self._save_model(model, name, metrics)
            self._register_model(model, metrics)
            
            # Update latest symlink
            latest_path = Path(self.config['MODEL_SAVE_DIR']) / 'latest'
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(self.model_dir.resolve())

    def _save_model(self, model, name, metrics):
        """Save model to disk"""
        model_path = self.model_dir / f'{name}.joblib'
        joblib.dump(model, model_path)
        
        metrics_path = self.model_dir / f'{name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved {name} model and metrics")

    def _register_model(self, model, metrics):
        """Register model in MLflow"""
        mlflow.set_tracking_uri(self.config.get('MLFLOW_TRACKING_URI', 'file:./mlruns'))
        
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_params({
                'model_type': type(model).__name__,
                'training_date': datetime.now().isoformat(),
                'retrain_reason': self.retrain_reason
            })
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(self.model_dir / 'reference_stats.json')
            
        logger.info(f"Registered model in MLflow: {mlflow.active_run().info.run_id}")

    def run(self):
        """Execute full training pipeline"""
        try:
            logger.info(f"Starting training pipeline (Retrain: {self.retrain})")
            best_model = self.train_models()
            
            if best_model:
                name, model, metrics = best_model
                logger.info(f"Training completed. Best model: {name} (F1: {metrics['f1']:.4f})")
                return True
            raise RuntimeError("No valid models trained")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Flag for retraining")
    parser.add_argument("--reason", default="scheduled", help="Reason for retraining")
    args = parser.parse_args()
    
    trainer = ModelTrainer(retrain=args.retrain, retrain_reason=args.reason)
    success = trainer.run()
    sys.exit(0 if success else 1)