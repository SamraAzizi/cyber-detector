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

    