import requests
import pandas as pd
import numpy as np
import shap
import joblib
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from backend.services.preprocessing import Preprocessor  # Your existing preprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardUtils:
    def __init__(
        self,
        model_path: str = "ml/models/rf_model.pkl",
        api_url: str = "http://backend:8000",
        preprocessor: Optional[Preprocessor] = None
    ):
        """
        Enhanced dashboard utilities with:
        - Preprocessor integration
        - Better error handling
        - Request retries
        - Caching for explanations
        """
        self.model = joblib.load(model_path)
        self.api_url = api_url.rstrip('/')
        self.preprocessor = preprocessor or Preprocessor()
        self.explainer = shap.TreeExplainer(self.model)
        self._feature_names = self.preprocessor.get_feature_names()
        self._session = requests.Session()  # For connection pooling
