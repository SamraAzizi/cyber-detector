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


        # Second split (train vs val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.val_size,
            stratify=y_train_val,
            random_state=self.random_state
        )
        
        # Save all splits
        self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _save_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save all data splits to disk"""
        X_train.to_csv(DATA_PATH/'X_train.csv', index=False)
        X_val.to_csv(DATA_PATH/'X_val.csv', index=False)
        X_test.to_csv(DATA_PATH/'X_test.csv', index=False)


        
        y_train.to_csv(DATA_PATH/'y_train.csv', index=False)
        y_val.to_csv(DATA_PATH/'y_val.csv', index=False)
        y_test.to_csv(DATA_PATH/'y_test.csv', index=False)

if __name__ == '__main__':
    # Example usage (would normally be called from training pipeline)
    from feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer()
    df = fe.load_data()
    df = fe.engineer_features(df)
    
    splitter = DataSplitter()
    splitter.split_data(df)
    print("Data splitting completed. Files saved to", DATA_PATH)