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


    


    def validate(self, package_dir: str) -> bool:
        """Run full validation suite"""
        print(colored("\nStarting Deployment Validation", 'cyan', attrs=['bold']))
        print(colored("=" * 50, 'cyan'))
        
        # 1. Test model loading
        if not self._test_model_loading(package_dir):
            self._log_test("Model Loading", 'failed', "Cannot continue validation")
            return False
            
        # 2. Validate schema
        self._test_schema_validation()
        
        # 3. Test example predictions
        self._test_example_predictions(package_dir)
        
        # 4. Performance benchmarks
        self._test_performance()
        
        # Summary
        print(colored("\nValidation Summary:", 'cyan', attrs=['bold']))
        print(colored(f"Passed: {self.results['passed']}", 'green'))
        print(colored(f"Warnings: {self.results['warnings']}", 'yellow'))
        print(colored(f"Failed: {self.results['failed']}", 'red'))
        
        return self.results['failed'] == 0


