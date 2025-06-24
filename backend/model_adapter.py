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




    def load_deployment(self, package_dir: str) -> bool:
        """
        Load a versioned model deployment package
        
        Args:
            package_dir: Path to the deployment package directory
            
        Returns:
            bool: True if loaded successfully
        """
        package_path = Path(package_dir)
        
        try:
            # Load metadata first
            with open(package_path / 'model_metadata.json') as f:
                self.metadata = json.load(f)
            
            # Load model using the backend's ModelLoader
            model_name = f"ThreatModel_{self.metadata['deployment_timestamp']}"
            self.current_model = self.loader.load_model(
                str(package_path / 'threat_model.pkl'),
                model_name
            )
            
            # Validate compatibility
            if not self._validate_compatibility():
                raise ValueError("Model compatibility check failed")
                
            print(colored("✅ Deployment package loaded successfully!", 'green'))
            return True
            
        except Exception as e:
            print(colored(f"❌ Deployment loading failed: {str(e)}", 'red'))
            logger.error(f"Deployment loading failed: {str(e)}")
            return False
        



    
    def _validate_compatibility(self) -> bool:
        """Validate model compatibility with the current backend"""
        if not self.metadata or not self.current_model:
            return False
            
        # Check backend version compatibility
        required_version = self.metadata.get('compatibility', {}).get('backend_version')
        if required_version != '2.0':
            print(colored(f"⚠️  Model requires backend version {required_version}", 'yellow'))
            
        # Check feature schema
        if not self._validate_feature_schema():
            return False
            
        return True
    
    


    def _validate_feature_schema(self) -> bool:
        """Validate model feature schema"""
        if 'input_schema' not in self.metadata:
            print(colored("⚠️  No input schema in metadata", 'yellow'))
            return True
            
        if not hasattr(self.current_model, 'feature_names_in_'):
            print(colored("⚠️  Model doesn't expose feature names", 'yellow'))
            return True
            
        # Compare metadata schema with actual model features
        metadata_features = set(self.metadata['input_schema'].keys())
        model_features = set(self.current_model.feature_names_in_)
        
        if metadata_features != model_features:
            print(colored("❌ Feature mismatch between metadata and model!", 'red'))
            print(colored(f"Metadata features: {metadata_features}", 'yellow'))
            print(colored(f"Model features: {model_features}", 'yellow'))
            return False
            
        return True
    
    



    def predict(self, input_data: Dict) -> Dict:
        """
        Make prediction with schema validation and enhanced output
        
        Args:
            input_data: Dictionary of feature values
            
        Returns:
            Dictionary containing:
            - prediction: The model prediction
            - confidence: Prediction confidence
            - metadata: Model metadata
            - timestamp: Prediction timestamp
        """
        if not self.current_model:
            raise ValueError("No model loaded for prediction")
            
        try:
            # Prepare features in correct order
            features = self._prepare_features(input_data)
            
            # Make prediction using the backend loader
            prediction, visualization = self.loader.predict(
                self.current_model,
                features,
                feature_names=self.metadata['input_schema'].keys(),
                visualize=True
            )
            
            # Prepare comprehensive output
            return {
                'prediction': float(prediction),
                'confidence': self._get_confidence(prediction),
                'model_version': self.metadata['deployment_timestamp'],
                'timestamp': datetime.now().isoformat(),
                'visualization': visualization,
                'metadata': {
                    'model_type': self.metadata['model_type'],
                    'performance': self.metadata.get('performance_metrics', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction error: {str(e)}")




    def _prepare_features(self, input_data: Dict) -> list:
        """Prepare features in correct order with validation"""
        if not self.metadata.get('input_schema'):
            return list(input_data.values())
            
        features = []
        for feat_name, feat_meta in self.metadata['input_schema'].items():
            if feat_name not in input_data:
                raise ValueError(f"Missing required feature: {feat_name}")
            features.append(float(input_data[feat_name]))
            
        return features
    


    
    def _get_confidence(self, prediction) -> float:
        """Calculate prediction confidence score"""
        # This can be enhanced based on model type
        return 1.0 if prediction in [0, 1] else 0.5
    


    
    
    def get_stats(self) -> Dict:
        """Get combined stats from loader and deployment info"""
        stats = self.loader.get_stats()
        stats.update({
            'deployment': {
                'version': self.metadata.get('deployment_timestamp'),
                'model_type': self.metadata.get('model_type'),
                'performance': self.metadata.get('performance_metrics', {})
            }
        })
        return stats
