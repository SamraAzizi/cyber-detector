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