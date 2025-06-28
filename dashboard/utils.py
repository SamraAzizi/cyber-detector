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