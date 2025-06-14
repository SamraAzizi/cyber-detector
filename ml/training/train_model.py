import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from pathlib import Path
import logging
from ml.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

def load_data(file_path):
    """Load dataset from CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the dataset."""
    # Handle missing values
    data = data.dropna()
    
    # Convert categorical data to numerical if needed
    # Example: data['protocol_type'] = data['protocol_type'].astype('category').cat.codes
    
    # Separate features and target
    X = data.drop(config['TARGET_COLUMN'], axis=1)
    y = data[config['TARGET_COLUMN']]
    
    return X, y

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=config['N_ESTIMATORS'],
        max_depth=config['MAX_DEPTH'],
        random_state=config['RANDOM_STATE']
    )

    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Save trained model to disk."""
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    try:
        # Load and preprocess data
        data = load_data(config['DATASET_PATH'])
        X, y = preprocess_data(data)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['TEST_SIZE'], 
            random_state=config['RANDOM_STATE']
        )