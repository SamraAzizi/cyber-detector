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
    


    def _apply_time_features(self, df):
        """Apply time feature engineering"""
        for feature in self.feature_config.get('time_features', []):
            if feature == 'hour':
                df['hour'] = df['timestamp'].dt.hour
            elif feature == 'day_of_week':
                df['day_of_week'] = df['timestamp'].dt.dayofweek
            elif feature == 'is_weekend':
                df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        return df
    

    
    
    def _apply_network_features(self, df):
        """Apply network feature engineering"""
        for feature in self.feature_config.get('network_features', []):
            if feature == 'bytes_ratio':
                df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
            elif feature == 'packet_rate':
                df['packet_rate'] = df['src_pkts'] / (df['duration'] + 0.001)
        return df
    


    def _apply_protocol_features(self, df):
        """Apply protocol feature engineering"""
        for proto in self.feature_config.get('protocols', []):
            proto_name = proto.split('_')[-1]  
            df[proto] = (df['protocol'] == proto_name).astype(int)
        return df
    
    def _get_numerical_cols(self):
        """Get list of numerical columns that need scaling"""
        
        return ['duration', 'src_bytes', 'dst_bytes']  
    
    def _ensure_columns(self, df):
        """Ensure all expected columns are present"""
        expected_cols = set()
        expected_cols.update(self.feature_config.get('time_features', []))
        expected_cols.update(self.feature_config.get('network_features', []))
        expected_cols.update(self.feature_config.get('protocols', []))
        
        
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        return df

    

    