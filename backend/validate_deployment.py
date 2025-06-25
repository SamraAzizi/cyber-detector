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
        for i, example in enumerate(examples[:3]):  # Test first 3 examples
            try:
                # Prepare input
                input_data = dict(zip(example['feature_names'], example['features']))
                
                # Make prediction
                result = self.adapter.predict(input_data)
                
                # Compare with expected
                predicted_class = result['prediction']
                expected_class = example.get('predicted_class')
                
                if expected_class is not None and predicted_class != expected_class:
                    self._log_test(f"Example {i+1}", 'failed', 
                                  f"Predicted {predicted_class}, expected {expected_class}")
                else:
                    passed += 1
                    
            except Exception as e:
                self._log_test(f"Example {i+1}", 'failed', str(e))
                
        if passed == len(examples[:3]):
            self._log_test("Example Predictions", 'passed', f"{passed}/{len(examples[:3])} passed")






    def _test_performance(self):
        """Run basic performance benchmarks"""
        if not self.adapter.metadata.get('input_schema'):
            self._log_test("Performance Benchmark", 'warning', "No schema for performance test")
            return
            
        try:
            import time
            test_input = {k: 0.0 for k in self.adapter.metadata['input_schema'].keys()}
            
            # Warmup
            self.adapter.predict(test_input)
            
            # Benchmark
            start = time.time()
            for _ in range(100):
                self.adapter.predict(test_input)
            elapsed = time.time() - start
            
        
        
            avg_latency = elapsed / 100
            self._log_test("Performance Benchmark", 'passed', 
                          f"Average latency: {avg_latency:.4f}s")
            
            if avg_latency > 0.1:
                self._log_test("Latency Warning", 'warning', 
                              "Latency >100ms may impact real-time performance")
                
        except Exception as e:
            self._log_test("Performance Benchmark", 'failed', str(e))




if __name__ == "__main__":
    validator = DeploymentValidator()
    success = validator.validate("ml/models/deployment_packages/latest")
    
    if not success:
        print(colored("\nDEPLOYMENT VALIDATION FAILED", 'red', attrs=['bold']))
        exit(1)
    else:
        print(colored("\nDEPLOYMENT VALIDATION PASSED", 'green', attrs=['bold']))
