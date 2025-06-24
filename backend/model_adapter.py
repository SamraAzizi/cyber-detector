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
    
    
