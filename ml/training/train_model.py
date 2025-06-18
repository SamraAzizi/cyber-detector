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