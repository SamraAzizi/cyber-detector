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
                



    def explain_prediction(self, raw_packet: Dict) -> Dict:
        """
        Enhanced SHAP explanation with:
        - Preprocessing integration
        - Feature name alignment
        - Error handling
        Returns:
            {
                "prediction": 0.92,
                "features": [
                    {
                        "name": "src_bytes",
                        "value": 1024,
                        "shap": 0.15,
                        "scaled_value": 1.2  # Added scaled value
                    },
                    ...
                ],
                "feature_importance": {
                    "src_bytes": 0.32,
                    ...
                }
            }
        """
        try:
            # Preprocess input (matches training exactly)
            processed = self.preprocessor.preprocess(raw_packet)
            
            # Get prediction probability
            proba = self.model.predict_proba(processed)[0][1]
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(processed)[1][0]
            
            # Get feature values (original + scaled)
            feature_values = processed[0]
            
            # Format for dashboard

            return {
                "prediction": float(proba),
                "features": [
                    {
                        "name": name,
                        "original_value": raw_packet.get(name.split('_')[0], 0),  # Get base feature
                        "scaled_value": float(val),
                        "shap": float(shap_val)
                    }
                    for name, val, shap_val in zip(
                        self._feature_names,
                        feature_values,
                        shap_values
                    )
                ],
                "feature_importance": self._get_global_importance()
            }
            
            
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}")
            return {
                "error": str(e),
                "prediction": 0.0,
                "features": []
            }



            
    def _get_global_importance(self) -> Dict[str, float]:
        """Get model-level feature importance for context"""
        if hasattr(self.model, 'feature_importances_'):
            return {
                name: float(imp)
                for name, imp in zip(
                    self._feature_names,
                    self.model.feature_importances_
                )
            }
        return {}
    

    

    def get_api_health(self) -> Dict:
        """Check if backend API is available"""
        try:
            response = self._session.get(
                f"{self.api_url}/health",
                timeout=2.0
            )
            return {
                "status": "healthy" if response.ok else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e)
            }
        

        

# Legacy function with deprecation warning
def get_predictions(text: str) -> Dict:
    logger.warning("Deprecated: Use DashboardUtils class instead")
    API_URL = "http://backend:8000/predict"
    
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=3.0
        )
        if response.ok:
            return response.json()
        return {"label": "API Error", "confidence": 0, "status_code": response.status_code}
    except Exception as e:
        return {"label": "Connection Error", "confidence": 0, "error": str(e)}


