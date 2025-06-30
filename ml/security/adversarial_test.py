from art.attacks.evasion import (
    FastGradientMethod,
    CarliniL2Method,
    DeepFool,
    ProjectedGradientDescent
)
from art.estimators.classification import SklearnClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
import json
import warnings
warnings.filterwarnings('ignore')  # Suppress ART warnings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialTester:

    def __init__(self, model_path: str, config_path: str = "ml/configs/security.json"):
        """
        Enhanced adversarial testing with:
        - Multiple attack types
        - Detailed reporting
        - Defense recommendations
        Args:
            model_path: Path to trained model (.pkl)
            config_path: Security testing parameters
        """
        self.model = joblib.load(model_path)
        self.classifier = SklearnClassifier(model=self.model)
        self.load_config(config_path)
        


    def load_config(self, config_path: str):
        """Load attack parameters from JSON config"""
        try:
            with open(config_path) as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.warning("No security config found, using defaults")
            self.config = {
                "attacks": {
                    "fgsm": {"eps": 0.1},
                    "carlini": {"confidence": 0.5},
                    "deepfool": {"max_iter": 50}
                },
                "thresholds": {
                    "min_accuracy": 0.7,
                    "max_success_rate": 0.3
                }
            }



    def run_all_tests(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str = "ml/security/reports"
    ) -> Dict[str, Dict]:
        """
        Execute all configured attack tests.
        Returns:
            Dictionary of attack reports
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        reports = {}
        
        for attack_name in self.config["attacks"]:
            report = self.run_test(
                attack_name=attack_name,
                X_val=X_val,
                y_val=y_val,
                output_dir=output_dir
            )
            reports[attack_name] = report
            
        self.generate_summary(reports, output_dir)
        return reports
