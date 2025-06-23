import pickle
import json
from pathlib import Path
from typing import Dict, Any
from termcolor import colored
from .model_loader import ModelLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelAdapter:
    """
    Adapter for the backend ModelLoader that adds deployment-specific features:
    - Versioned model loading
    - Schema validation
    - Automatic metadata handling
    - Backward compatibility
    """
    
    def __init__(self):
        self.loader = ModelLoader()
        self.current_model = None
        self.metadata = None
        self._print_welcome()
        
    def _print_welcome(self):
        """Display deployment-specific welcome message"""
        print(colored("\n" + "="*60, 'magenta'))
        print(colored("🚀 MODEL DEPLOYMENT ADAPTER v1.0", 'cyan', attrs=['bold']))
        print(colored("Advanced Deployment Integration Layer", 'yellow'))
        print(colored("="*60 + "\n", 'magenta'))