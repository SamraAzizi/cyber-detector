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
    


    def run_test(
        self,
        attack_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_dir: str = None
    ) -> Dict:
        """
        Execute single attack test with:
        - Attack generation
        - Robustness metrics
        - Example adversarial samples
        """
        try:
            # Initialize attack
            attack = self._init_attack(attack_name)
            
            # Generate adversarial examples
            X_adv = attack.generate(X_val)
            y_pred = self.classifier.predict(X_adv)
            y_pred_labels = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_val, y_pred_labels),
                "precision": precision_score(y_val, y_pred_labels, zero_division=0),
                "recall": recall_score(y_val, y_pred_labels, zero_division=0),
                "success_rate": 1 - accuracy_score(y_val, y_pred_labels),
                "confusion_matrix": confusion_matrix(y_val, y_pred_labels).tolist()
            }
            
            # Save adversarial examples
            if output_dir:
                self._save_attack_samples(
                    X_adv=X_adv,
                    y_pred=y_pred_labels,
                    attack_name=attack_name,
                    output_dir=output_dir
                )
                
            # Evaluate against thresholds
            metrics["passed"] = (
                metrics["accuracy"] >= self.config["thresholds"]["min_accuracy"] and
                metrics["success_rate"] <= self.config["thresholds"]["max_success_rate"]
            )
            
            logger.info(
                f"{attack_name.upper()} Test - "
                f"Accuracy: {metrics['accuracy']:.2%}, "
                f"Success Rate: {metrics['success_rate']:.2%}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"{attack_name} attack failed: {str(e)}")
            return {"error": str(e)}



    def _init_attack(self, attack_name: str):
        """Initialize attack based on config"""
        params = self.config["attacks"][attack_name]
        
        if attack_name == "fgsm":
            return FastGradientMethod(
                estimator=self.classifier,
                eps=params["eps"]
            )
        elif attack_name == "carlini":
            return CarliniL2Method(
                classifier=self.classifier,
                confidence=params["confidence"]
            )
        elif attack_name == "deepfool":
            return DeepFool(
                classifier=self.classifier,
                max_iter=params["max_iter"]
            )
        elif attack_name == "pgd":
            return ProjectedGradientDescent(
                estimator=self.classifier,
                eps=params["eps"],
                eps_step=params["eps_step"]
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_name}")
