import requests
import pandas as pd
import numpy as np
import shap
import joblib
from typing import Dict, List, Any

class DashboardUtils:

    def __init__(self, model_path: str = "ml/models/rf_model.pkl", api_url: str = "http://localhost:8000"):
        self.model = joblib.load(model_path)
        self.api_url = api_url
        self.explainer = shap.TreeExplainer(self.model)



    def fetch_live_threats(self, packets: List[Dict]) -> List[Dict]:
        """Get predictions from backend API."""
        response = requests.post(
            f"{self.api_url}/predict_batch",
            json={"packets": packets}
        )
        return response.json()["threats"]
    


    def explain_prediction(self, packet: Dict) -> Dict:
        """
        Generate SHAP explanation for a single prediction.
        Returns:
            {
                "prediction": 0.92, 
                "features": [
                    {"name": "src_bytes", "value": 1024, "shap": 0.15},
                    {"name": "protocol_type_tcp", "value": 1, "shap": 0.07}
                ]
            }
        """
        # Get prediction probability
        proba = self.model.predict_proba([list(packet.values())])[0][1]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values([list(packet.values())])[1][0]
        
        # Format for dashboard
        return {
            "prediction": float(proba),
            "features": [
                {
                    "name": name,
                    "value": val,
                    "shap": float(shap_val)
                }
                for name, val, shap_val in zip(
                    packet.keys(),
                    packet.values(),
                    shap_values
                )
            ]
        }
