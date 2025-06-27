import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from pathlib import Path
import time
from typing import List


class ModelOptimizer:
    @staticmethod
    def convert_to_onnx(model_path: str, output_dir: str, n_features: int):
        """Convert sklearn model to ONNX format (3-5x faster inference)."""
        model = joblib.load(model_path)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        output_path = Path(output_dir) / f"{Path(model_path).stem}.onnx"
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to {output_path}")
