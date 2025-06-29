import numpy as np
from art.attacks.evasion import FastGradientMethod, CarliniL2Method
from art.estimators.classification import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from typing import Tuple



class AdversarialTester:
    
    def __init__(self, model_path: str = "ml/models/rf_model.pkl"):
        """
        Test model against evasion attacks using ART.
        Requires: pip install adversarial-robustness-toolbox
        """
        self.model = joblib.load(model_path)
        self.classifier = SklearnClassifier(model=self.model)
        self.logger = logging.getLogger(__name__)


    def test_evasion(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        attack_type: str = "fgsm"
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate model robustness against attacks.
        Returns:
            - Attack success rate
            - Adversarial examples
        """
        try:
            if attack_type == "fgsm":
                attack = FastGradientMethod(self.classifier, eps=0.1)
            elif attack_type == "carlini":
                attack = CarliniL2Method(self.classifier, targeted=False)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            X_adv = attack.generate(X_test)
            adv_predictions = self.model.predict(X_adv)
            success_rate = np.mean(adv_predictions != y_test)

            self.logger.info(f"Attack success rate ({attack_type}): {success_rate:.2%}")
            return success_rate, X_adv

        except Exception as e:
            self.logger.error(f"Adversarial testing failed: {str(e)}")
            raise

# Usage:
# tester = AdversarialTester()
# success_rate, examples = tester.test_evasion(X_test, y_test, "fgsm")

