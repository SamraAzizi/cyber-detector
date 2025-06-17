import pandas as np
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