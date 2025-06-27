import mlflow
import pickle
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        mlflow.set_tracking_uri("file:./ml/registry")
        self.experiment = mlflow.set_experiment("CyberShield")
    
    def log_model(self, model, metrics: dict):
        """Register a new model version"""
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log metadata
            mlflow.log_params({
                'training_date': datetime.now().isoformat(),
                'model_type': type(model).__name__
            })
            
            logger.info(f"Logged model: {mlflow.active_run().info.run_id}")