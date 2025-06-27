import subprocess
from registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)

def trigger_retraining(reason: str):
    """Execute full retraining pipeline"""
    try:
        # 1. Run training script
        subprocess.run([
            "python", "ml/training/train_model.py",
            "--retrain", "true",
            "--reason", reason
        ], check=True)
        
        # 2. Package new model
        subprocess.run([
            "python", "ml/training/model_packager.py"
        ], check=True)
        
        logger.info("Retraining pipeline completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {str(e)}")
        return False