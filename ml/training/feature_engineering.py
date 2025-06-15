import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# Constants aligned with project structure
DATA_PATH = Path('ml/datasets/cyber_threats.csv')
MODEL_DIR = Path('ml/models/')
MODEL_DIR.mkdir(exist_ok=True)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_config = {}  # To store feature engineering parameters
        
    def load_data(self):
        """Load and preprocess raw data"""
        df = pd.read_csv(DATA_PATH)
        
        # Convert timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def engineer_features(self, df):
        """Apply all feature engineering steps"""
        # Time-based features
        if 'timestamp' in df.columns:
            df = self._add_time_features(df)