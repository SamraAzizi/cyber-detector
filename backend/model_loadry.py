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
            
