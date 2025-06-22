import requests

API_URL = "http://backend:8000/predict"  # Update this if running locally or via docker-compose

def get_predictions(text):
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            return {"label": "Error", "confidence": "0"}
    except Exception as e:
        return {"label": "API Error", "confidence": "0"}

