import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime



class Prepreocessor:
    def __init__(self):
        # Load feature engineering configuration
        self.feature_config = self._load_feature_config()
        self.scaler = joblib.load(Path('ml/models/scaler.pkl'))


    
    def _load_feature_config(self):
        """
        Load feature engineering config from training
        """
        with open(Path('ml/models/feature_config.json')) as f:
            return json.load(f)
        

    
    def preprocess(self, input_data):
        """Preprocess incoming data for inference"""
        
        if not isinstance(input_data, pd.DataFrame):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = self._apply_time_features(df)
        
        df = self._apply_network_features(df)
        df = self._apply_protocol_features(df)
        
        
        numerical_cols = self._get_numerical_cols()
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        
        df = self._ensure_columns(df)
        
        return df

    

    