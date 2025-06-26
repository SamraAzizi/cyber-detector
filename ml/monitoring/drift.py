import pandas as pd
from scipy.stats import ks_2samp
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DriftDetector:

    def __init__(self):
        self.reference = self._load_reference()
    



    def _load_reference(self):
        """Load training data stats from model package"""
        try:
            with open('ml/models/deployment_packages/latest/reference_stats.json') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Reference stats not found. Generate during training!")
            return None




    def check_drift(self, features: dict):
        """Compare production features to training distribution"""
        if not self.reference:
            return None
            
        drift_results = {}
