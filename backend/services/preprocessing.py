import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Union, Any
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, model_dir: str = "ml/models"):
        """
        Enhanced preprocessor with:
        - Type hints for better IDE support
        - Input validation
        - Feature documentation
        - Caching for frequent features
        """
        self.model_dir = Path(model_dir)
        self.feature_config = self._load_feature_config()
        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        self._initialize_feature_cache()
        
        # For dashboard explanations
        self.feature_names = self._get_feature_names()

    def _load_feature_config(self) -> Dict:
        """Load feature config with error handling"""
        try:
            with open(self.model_dir / 'feature_config.json') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("feature_config.json not found - using defaults")
            return {
                'time_features': ['hour', 'day_of_week'],
                'network_features': ['bytes_ratio', 'packet_rate'],
                'protocols': ['proto_tcp', 'proto_udp']
            }

    def _initialize_feature_cache(self):
        """Cache frequently used feature definitions"""
        self.numerical_cols = ['duration', 'src_bytes', 'dst_bytes']
        self.protocol_mapping = {
            'tcp': 'proto_tcp',
            'udp': 'proto_udp',
            'icmp': 'proto_icmp'
        }

    def _get_feature_names(self) -> list:
        """Generate complete feature names for SHAP explanations"""
        features = []
        features.extend(self.numerical_cols)
        
        # Add time features
        if 'hour' in self.feature_config.get('time_features', []):
            features.append('hour')
        if 'day_of_week' in self.feature_config.get('time_features', []):
            features.append('day_of_week')
            
        # Add network features
        features.extend(self.feature_config.get('network_features', []))
        
        # Add protocol dummies
        features.extend(self.feature_config.get('protocols', []))
        
        return features
    



    def preprocess(self, input_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Enhanced preprocessing with:
        - Input type flexibility (dict or DataFrame)
        - Input validation
        - Memory efficiency
        """
        if not isinstance(input_data, pd.DataFrame):
            try:
                df = pd.DataFrame([input_data])
            except Exception as e:
                logger.error(f"Input conversion failed: {str(e)}")
                raise ValueError("Input must be dict or DataFrame")
        else:
            df = input_data.copy()

        # Validate required columns
        self._validate_input(df)

        # Feature engineering pipeline
        try:
            if 'timestamp' in df.columns:
                df = self._apply_time_features(df)
            
            df = self._apply_network_features(df)
            df = self._apply_protocol_features(df)
            df = self._scale_features(df)
            df = self._ensure_columns(df)
            
            return df[self.feature_names].values.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise RuntimeError("Feature engineering error")
        

    
    def _validate_input(self, df: pd.DataFrame):
        """Validate required columns and types"""
        required = {
            'timestamp': 'datetime64[ns]',
            'protocol': 'object',
            'src_bytes': 'int64',
            'dst_bytes': 'int64',
            'duration': 'float64'
        }
        
        for col, dtype in required.items():
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            if not np.issubdtype(df[col].dtype, np.dtype(dtype)):
                raise ValueError(f"Column {col} must be {dtype}")
            

            

    def _apply_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized time feature extraction"""
        ts = pd.to_datetime(df['timestamp'])
        for feature in self.feature_config.get('time_features', []):
            if feature == 'hour':
                df['hour'] = ts.dt.hour
            elif feature == 'day_of_week':
                df['day_of_week'] = ts.dt.dayofweek
            elif feature == 'is_weekend':
                df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
        return df


    
    