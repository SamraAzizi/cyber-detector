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

        
        # Network traffic features
        df = self._add_network_features(df)
        
        # Protocol-specific features
        df = self._add_protocol_features(df)
        
        # Save feature config for inference
        self._save_feature_config()
        
        return df
    
    def _add_time_features(self, df):
        """Create time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        
        # Store which time features were created
        self.feature_config['time_features'] = ['hour', 'day_of_week', 'is_weekend']
        return df
    
    def _add_network_features(self, df):
        """Create network traffic features"""
        # Example features - adjust based on actual columns
        if {'src_bytes', 'dst_bytes'}.issubset(df.columns):
            df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)  # +1 to avoid div by zero
            
        if {'duration', 'src_pkts'}.issubset(df.columns):
            df['packet_rate'] = df['src_pkts'] / (df['duration'] + 0.001)
        
        self.feature_config['network_features'] = ['bytes_ratio', 'packet_rate']
        return df
