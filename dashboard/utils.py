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




    def fetch_live_threats(self, packets: List[Dict], max_retries: int = 3) -> List[Dict]:
        """
        Get batch predictions from backend API with:
        - Retry logic
        - Timeout handling
        - Data validation
        """
        url = f"{self.api_url}/predict_batch"
        
        for attempt in range(max_retries):
            try:
                response = self._session.post(
                    url,
                    json={"packets": packets},
                    timeout=5.0
                )
                response.raise_for_status()
                
                # Validate response format
                result = response.json()
                if not isinstance(result.get("threats"), list):
                    raise ValueError("Invalid API response format")
                    
                return result["threats"]
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for API request")
                    return [{"error": str(e)}] * len(packets)
                
