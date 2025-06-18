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