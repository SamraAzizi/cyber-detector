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
