import pickle
import numpy as np
import time
from datetime import datetime
from termcolor import colored
import logging
from typing import Optional, Tuple, Any
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from sklearn.metrics import classification_report
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelLoader:
    """
    Advanced Model Loader with Performance Tracking and Visualization
    
    Features:
    - Beautiful colored console output
    - Detailed logging
    - Prediction performance metrics
    - Model validation
    - Prediction visualization
    - Memory optimization
    """
    
    def __init__(self):
        self.loaded_models = {}
        self.prediction_stats = {}
        self.logger = self._setup_logging()
        self._print_welcome()
        
    def _print_welcome(self):
        """Display beautiful welcome message"""
        print(colored("\n" + "="*60, 'blue'))
        print(colored("✨ AI MODEL LOADER v2.0", 'cyan', attrs=['bold']))
        print(colored("Advanced Threat Detection Engine", 'yellow'))
        print(colored("="*60 + "\n", 'blue'))
        
    def _setup_logging(self):
        """Configure advanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_loader.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('ModelLoader')
    
    def load_model(self, model_path: str, model_name: str = None) -> Optional[Any]:
        """
        Load a trained ML model from file with enhanced features
        
        Args:
            model_path: Path to the model file
            model_name: Friendly name for the model (optional)
            
        Returns:
            Loaded model object or None if failed
        """
        model_name = model_name or model_path.split('/')[-1]
        
        try:
            start_time = time.time()
            
            # Memory-efficient loading for large models
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            load_time = time.time() - start_time
            self.loaded_models[model_name] = {
                'model': model,
                'load_time': load_time,
                'path': model_path,
                'predictions': 0
            }
            
            # Print success message
            msg = (f"✅ Successfully loaded model '{model_name}' "
                  f"({load_time:.2f}s) | Features: {self._get_model_features(model)}")
            print(colored(msg, 'green'))
            self.logger.info(msg)
            
            return model
            
        except Exception as e:
            error_msg = f"❌ Failed to load model '{model_name}': {str(e)}"
            print(colored(error_msg, 'red'))
            self.logger.error(error_msg)
            return None
    
    def _get_model_features(self, model) -> str:
        """Get model features information if available"""
        try:
            if hasattr(model, 'n_features_in_'):
                return f"{model.n_features_in_} features"
            if hasattr(model, 'coef_'):
                return f"{len(model.coef_[0])} features"
            return "unknown features"
        except:
            return "unknown features"
    
    def predict(self, model: Any, features: list, 
                feature_names: list = None, 
                visualize: bool = False) -> Tuple[float, Optional[str]]:
        """
        Make prediction using loaded model with advanced features
        
        Args:
            model: Loaded model object
            features: Input features for prediction
            feature_names: Names of features for visualization
            visualize: Whether to generate visualization
            
        Returns:
            Tuple of (prediction, visualization_base64_or_None)
        """
        if model is None:
            error_msg = "Model not loaded - prediction aborted"
            print(colored(error_msg, 'red'))
            self.logger.error(error_msg)
            return -1, None
            
        try:
            start_time = time.time()
            features_array = np.array(features).reshape(1, -1)
            
            # Get prediction and probability if available
            prediction = model.predict(features_array)[0]
            proba = None
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_array)[0]
                confidence = max(proba)
            else:
                confidence = 1.0
                
            pred_time = time.time() - start_time
            
