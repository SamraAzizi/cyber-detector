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



    @staticmethod
    def benchmark(model_path: str, test_data: List[Dict], n_runs: int = 1000):
        """Compare inference speed between pickle and ONNX models."""
        # Load original model
        model = joblib.load(model_path)
        
        # Time pickle model
        start = time.time()
        for _ in range(n_runs):
            model.predict([list(test_data[0].values())])
        pickle_time = time.time() - start

        # Time ONNX model (if exists)
        onnx_path = model_path.replace(".pkl", ".onnx")
        if Path(onnx_path).exists():
            from onnxruntime import InferenceSession
            sess = InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            
            start = time.time()
            for _ in range(n_runs):
                sess.run(
                    None, 
                    {input_name: np.array([list(test_data[0].values())], dtype=np.float32)}
                )
            onnx_time = time.time() - start
            print(f"Pickle: {pickle_time:.2f}s | ONNX: {onnx_time:.2f}s ({pickle_time/onnx_time:.1f}x faster)")
        else:
            print(f"Pickle model: {n_runs} predictions in {pickle_time:.2f}s")



