import pickle
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
from termcolor import colored
import pandas as pd

logger = logging.getLogger(__name__)

class ModelPackager:
    def __init__(self):
        self.config = load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.package_dir = Path(self.config['MODEL_SAVE_DIR']) / 'deployment_packages' / self.timestamp
        self.package_dir.mkdir(parents=True, exist_ok=True)
        
    def _print_status(self, message, status='info'):
        """Colorful status messages matching the backend style"""
        color = {
            'info': 'blue',
            'success': 'green',
            'warning': 'yellow',
            'error': 'red'
        }.get(status, 'white')
        
        print(colored(f"• {message}", color))
        getattr(logger, status)(message)

    def package_for_backend(self, model_path, metadata=None):
        """Create a deployment package compatible with the backend ModelLoader"""
        try:
            self._print_status("Starting model packaging...")
            
            # Load the trained model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Generate comprehensive metadata
            metadata = self._generate_metadata(model, metadata)
            
            # Save model in backend-compatible format
            self._save_model(model, metadata)
            
            # Save additional artifacts
            self._save_artifacts(model, metadata)
            
            self._print_status("Packaging completed successfully!", 'success')
            return self.package_dir
            
        except Exception as e:
            self._print_status(f"Packaging failed: {str(e)}", 'error')
            raise

    def _generate_metadata(self, model, existing_metadata=None):
        """Generate rich metadata for the backend"""
        metadata = {
            'model_type': type(model).__name__,
            'deployment_timestamp': self.timestamp,
            'input_schema': self._get_input_schema(model),
            'output_classes': self._get_output_classes(model),
            'performance_metrics': existing_metadata.get('metrics', {}) if existing_metadata else {},
            'compatibility': {
                'backend_version': '2.0',
                'loader_class': 'ModelLoader'
            }
        }
        
        # Add model-specific information
        if hasattr(model, 'feature_importances_'):
            metadata['feature_importances'] = dict(zip(
                model.feature_names_in_,
                model.feature_importances_
            ))
            
        if existing_metadata:
            metadata.update(existing_metadata)
            
        return metadata

    def _get_input_schema(self, model):
        """Generate input feature schema"""
        schema = {}
        
        if hasattr(model, 'feature_names_in_'):
            for i, name in enumerate(model.feature_names_in_):
                schema[name] = {
                    'position': i,
                    'dtype': 'float32',  # Default, can be enhanced
                    'description': f'Feature {name}'
                }
        return schema

    def _get_output_classes(self, model):
        """Get output classes information"""
        if hasattr(model, 'classes_'):
            return {
                str(cls): {'index': i, 'type': type(cls).__name__}
                for i, cls in enumerate(model.classes_)
            }
        return {'0': 'binary'}

    def _save_model(self, model, metadata):
        """Save model in backend-compatible format"""
        model_path = self.package_dir / 'threat_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        self._print_status(f"Model saved to {model_path}")

    def _save_artifacts(self, model, metadata):
        """Save additional deployment artifacts"""
        # Save metadata
        metadata_path = self.package_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save example predictions
        self._save_examples(model, metadata)
        
        self._print_status(f"Artifacts saved to {self.package_dir}")

    def _save_examples(self, model, metadata):
        """Save example predictions for validation"""
        if not hasattr(model, 'feature_names_in_'):
            return
            
        examples = []
        np.random.seed(42)
        
        for _ in range(5):
            example = {
                'features': np.random.rand(len(model.feature_names_in_)).tolist(),
                'feature_names': model.feature_names_in_.tolist()
            }
            
            try:
                prediction = model.predict_proba([example['features']])[0].tolist()
                example['prediction'] = prediction
                example['predicted_class'] = int(np.argmax(prediction))
            except:
                continue
                
            examples.append(example)
        
        examples_path = self.package_dir / 'example_predictions.json'
        with open(examples_path, 'w') as f:
            json.dump(examples, f, indent=2)

if __name__ == "__main__":
    packager = ModelPackager()
    packager.package_for_backend(
        model_path="ml/models/latest/model.joblib",
        metadata={
            'metrics': {
                'accuracy': 0.982,
                'f1_score': 0.963,
                'precision': 0.971,
                'recall': 0.955
            },
            'training_info': {
                'dataset': 'cyber_threats_v3',
                'train_date': '2023-12-15'
            }
        }
    )