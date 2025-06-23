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
