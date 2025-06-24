import json
from pathlib import Path
from termcolor import colored
from .model_adapter import ModelAdapter
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DeploymentValidator:
    """
    Comprehensive deployment validation that checks:
    - Model loading
    - Schema compatibility
    - Prediction consistency
    - Performance benchmarks
    """
    
    def __init__(self):
        self.adapter = ModelAdapter()
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }




    def _log_test(self, name, status, message=None):
        """Log test result with colorful output"""
        color = {
            'passed': 'green',
            'failed': 'red',
            'warning': 'yellow'
        }.get(status, 'white')
        
        symbol = {
            'passed': '✓',
            'failed': '✗',
            'warning': '!'
        }.get(status, '?')
        
        log_entry = {
            'test': name,
            'status': status,
            'message': message
        }
        
        self.results['details'].append(log_entry)
        self.results[status] += 1
        
        print(colored(f"{symbol} {name}: {status.upper()}", color))
        if message:
            print(colored(f"   → {message}", color))