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





    def _test_model_loading(self, package_dir: str) -> bool:
        """Test that the model loads successfully"""
        try:
            success = self.adapter.load_deployment(package_dir)
            if success:
                self._log_test("Model Loading", 'passed')
                return True
            else:
                self._log_test("Model Loading", 'failed')
                return False
        except Exception as e:
            self._log_test("Model Loading", 'failed', str(e))
            return False
    



    def _test_schema_validation(self):
        """Validate the model's input schema"""
        if not self.adapter.metadata.get('input_schema'):
            self._log_test("Schema Validation", 'warning', "No input schema in metadata")
            return
            
        try:
            # Test with incomplete input
            test_input = {k: 0.0 for k in list(self.adapter.metadata['input_schema'].keys())[:3]}
            try:
                self.adapter.predict(test_input)
                self._log_test("Schema Validation", 'failed', "Accepted incomplete input")
            except ValueError as e:
                self._log_test("Schema Validation", 'passed', "Properly rejected incomplete input")
                
            # Test with full input
            test_input = {k: 0.0 for k in self.adapter.metadata['input_schema'].keys()}
            self.adapter.predict(test_input)
            self._log_test("Full Input Validation", 'passed')
            
        except Exception as e:
            self._log_test("Schema Validation", 'failed', str(e))
    



    def _test_example_predictions(self, package_dir: str):
        """Validate against example predictions from training"""
        examples_path = Path(package_dir) / 'example_predictions.json'
        if not examples_path.exists():
            self._log_test("Example Predictions", 'warning', "No example predictions found")
            return
            
        with open(examples_path) as f:
            examples = json.load(f)
            
        passed = 0

