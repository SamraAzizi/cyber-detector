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
        
