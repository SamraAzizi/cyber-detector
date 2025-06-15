import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

# Constants aligned with project structure
DATA_PATH = Path('ml/datasets/processed/')
DATA_PATH.mkdir(exist_ok=True)

class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.25, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split_data(self, df, target_col='attack_type'):
        """Split data into train, validation, and test sets"""
        # Initial split (train+val vs test)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            stratify=y,
            random_state=self.random_state
        )