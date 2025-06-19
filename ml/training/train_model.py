import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import sys
from ml.utils import load_config  # Assuming you have this utility

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
        self.model = None
        self.metrics = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        


        
    def load_data(self):
        """Load and validate dataset"""
        try:
            logger.info(f"Loading data from {self.config['DATASET_PATH']}")
            data = pd.read_csv(self.config['DATASET_PATH'])
            
            # Validate required columns exist
            required_cols = self.config.get('REQUIRED_COLUMNS', [])
            if not set(required_cols).issubset(data.columns):
                missing = set(required_cols) - set(data.columns)
                raise ValueError(f"Missing required columns: {missing}")
                
            logger.info(f"Data shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            sys.exit(1)
        

    
    def preprocess_data(self, data):
        """Handle data preprocessing"""
        logger.info("Preprocessing data...")
        
        # Drop duplicates
        initial_size = len(data)
        data = data.drop_duplicates()
        logger.info(f"Removed {initial_size - len(data)} duplicates")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Feature engineering could be added here
        # Or imported from feature_engineering.py
        
        # Separate features and target
        X = data.drop(self.config['TARGET_COLUMN'], axis=1)
        y = data[self.config['TARGET_COLUMN']]
        
        return X, y
    


    def _handle_missing_values(self, data):
        """Custom missing value handling"""
        # Example strategy - adjust based on your data
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            logger.warning(f"Columns with missing values: {missing_cols}")
            
            # Numerical columns: median imputation
            num_cols = data.select_dtypes(include=np.number).columns
            num_cols_missing = list(set(num_cols) & set(missing_cols))
            
            for col in num_cols_missing:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
            
            # Categorical columns: mode imputation
            cat_cols = data.select_dtypes(exclude=np.number).columns
            cat_cols_missing = list(set(cat_cols) & set(missing_cols))
            
            for col in cat_cols_missing:
                mode_val = data[col].mode()[0]
                data[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        return data
    


    def train_model(self, X_train, y_train):
        """Model training with cross-validation"""
        logger.info("Training Random Forest model...")
        
        try:
            model = RandomForestClassifier(
                n_estimators=self.config.get('N_ESTIMATORS', 100),
                max_depth=self.config.get('MAX_DEPTH', None),
                class_weight=self.config.get('CLASS_WEIGHT', 'balanced'),
                random_state=self.config['RANDOM_STATE'],
                n_jobs=-1  # Use all available cores
            )
            
            model.fit(X_train, y_train)
            logger.info("Model training completed")
            return model
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            sys.exit(1)

            

    

    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else None,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Log important metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    



    def save_artifacts(self, model, metrics):
        """Save model and metrics"""
        artifacts_dir = Path('ml/models') / self.timestamp
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / 'model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = artifacts_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(model.feature_names_in_, model.feature_importances_))
            importance_path = artifacts_dir / 'feature_importance.json'
            with open(importance_path, 'w') as f:
                json.dump(importance, f, indent=2)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Update latest model reference
        latest_path = Path('ml/models/latest')
        latest_path.unlink(missing_ok=True)
        latest_path.symlink_to(artifacts_dir.resolve())
        logger.info(f"Updated latest model symlink to {artifacts_dir}")




    def run(self):
        """Execute full training pipeline"""
        try:
            logger.info("Starting training pipeline")
            
            # 1. Load data
            data = self.load_data()
            
            # 2. Preprocess
            X, y = self.preprocess_data(data)
            
            # 3. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['TEST_SIZE'],
                random_state=self.config['RANDOM_STATE'],
                stratify=y if self.config.get('STRATIFY_SPLIT', True) else None
            )
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # 4. Train model
            self.model = self.train_model(X_train, y_train)
            

            