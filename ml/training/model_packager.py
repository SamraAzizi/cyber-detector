import joblib
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import logging


logger = logging.getLogger(__name__)



class ModelPackager:

    def __init__(self, model_dir=None):
        self.config = load_config()
        self.model_dir = Path(model_dir) if model_dir else Path(self.config['MODEL_SAVE_DIR'])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.package_dir = self.model_dir / 'deployment_packages' / self.timestamp
        self.package_dir.mkdir(parents=True, exist_ok=True)




    def package_model(self, model_path, metadata=None):
        """Package model for deployment with all necessary artifacts"""
        try:
            # Load trained model
            model = joblib.load(model_path)
            
            # Generate default metadata if not provided
            if metadata is None:
                metadata = {
                    'model_type': type(model).__name__,
                    'timestamp': self.timestamp,
                    'input_features': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else [],
                    'output_classes': model.classes_.tolist() if hasattr(model, 'classes_') else []
                }
            
            # 1. Save as pickle (for Python backend)
            pickle_path = self.package_dir / 'model.pkl'
            joblib.dump(model, pickle_path)
            
            # 2. Convert to ONNX (for cross-platform deployment)
            self._convert_to_onnx(model, metadata)
            
            # 3. Save metadata
            metadata_path = self.package_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 4. Save feature engineering config
            self._copy_feature_config()
            
            logger.info(f"Model packaged successfully at {self.package_dir}")
            return self.package_dir
            
        except Exception as e:
            logger.error(f"Model packaging failed: {str(e)}")
            raise




    def _convert_to_onnx(self, model, metadata):
        """Convert sklearn model to ONNX format"""
        try:
            # Prepare initial type (adjust based on your feature count)
            n_features = len(metadata['input_features']) if metadata['input_features'] else 1
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12
            )
            
            # Save
            onnx_path = self.package_dir / 'model.onnx'
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
                
            logger.info(f"ONNX model saved to {onnx_path}")
            
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {str(e)}")
            raise




    def _copy_feature_config(self):
        """Copy feature engineering configuration"""
        try:
            feature_config = self.model_dir / 'feature_config.json'
            if feature_config.exists():
                destination = self.package_dir / 'feature_config.json'
                destination.write_text(feature_config.read_text())
        except Exception as e:
            logger.warning(f"Could not copy feature config: {str(e)}")




    def create_deployment_package(self):
        """Create deployment package from latest model"""
        latest_model = self.model_dir / 'latest' / 'model.joblib'
        
        