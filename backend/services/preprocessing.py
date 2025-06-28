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

    