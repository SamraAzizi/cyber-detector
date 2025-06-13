from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from model_loader import load_model, predict
from pydantic import BaseModel
from typing import List, Optional
import time
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CyberShield API",
    description="""**Advanced Threat Detection & Anomaly Modeling System** 🔍🛡️
    
    This API provides real-time cybersecurity threat detection using machine learning models trained on:
    - CICIDS 2017 dataset
    - NSL-KDD dataset
    
    ## Features:
    - Real-time threat scoring
    - Anomaly detection
    - Model health monitoring
    - Historical analysis""",
    version="1.0.0",
    contact={
        "name": "CyberSecurity Team",
        "email": "security@cybershield.ai",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Customize the Swagger UI
    openapi_schema["info"]["x-logo"] = {
        "url": "https://i.imgur.com/JZYkY3E.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup - with better error handling
try:
    threat_model = load_model('models/threat_detection_model.pkl')
    anomaly_model = load_model('models/anomaly_model.pkl')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    threat_model = None
    anomaly_model = None

# Enhanced data models
class NetworkData(BaseModel):
    """Network traffic data for analysis"""
    features: List[float]
    timestamp: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol: Optional[str] = None

class DetectionResult(BaseModel):
    """Result of threat/anomaly detection"""
    request_id: str
    timestamp: str
    threat_level: float
    anomaly_score: float
    is_threat: bool
    is_anomaly: bool
    confidence: float
    model_version: str = "1.0.0"

class HealthCheck(BaseModel):
    """System health status"""
    status: str
    models: dict
    uptime: float
    timestamp: str

# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_time = "{0:.2f}".format(process_time)
    logger.info(f"Request {request_id} completed: {formatted_time}ms")
    
    return response

# Custom docs endpoint for better Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1}
    )

# Health check endpoint
@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    """Check system health and model status"""
    return {
        "status": "OK",
        "models": {
            "threat_model": "loaded" if threat_model else "error",
            "anomaly_model": "loaded" if anomaly_model else "error"
        },
        "uptime": time.time() - app.startup_time,
        "timestamp": datetime.utcnow().isoformat()
    }

# Enhanced detection endpoints
@app.post("/detect-threat", response_model=DetectionResult, tags=["Detection"])
async def detect_threat(data: NetworkData):
    """Detect cybersecurity threats in network traffic
    
    - **features**: Array of network traffic features
    - **timestamp**: Optional timestamp of the event
    - Returns: Detailed threat analysis
    """
    if not threat_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Threat detection model not available"
        )
    
    try:
        prediction = predict(threat_model, data.features)
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
        
        return {
            "request_id": str(uuid.uuid4()),
            "timestamp": data.timestamp or datetime.utcnow().isoformat(),
            "threat_level": float(prediction),
            "anomaly_score": 0.0,
            "is_threat": prediction > 0.7,
            "is_anomaly": False,
            "confidence": float(confidence),
        }
    except Exception as e:
        logger.error(f"Threat detection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input data format"
        )

@app.post("/detect-anomaly", response_model=DetectionResult, tags=["Detection"])
async def detect_anomaly(data: NetworkData):
    """Detect anomalies in network traffic patterns
    
    - **features**: Array of network traffic features
    - **timestamp**: Optional timestamp of the event
    - Returns: Detailed anomaly analysis
    """
    if not anomaly_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anomaly detection model not available"
        )
    
    try:
        prediction = predict(anomaly_model, data.features)
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
        
        return {
            "request_id": str(uuid.uuid4()),
            "timestamp": data.timestamp or datetime.utcnow().isoformat(),
            "threat_level": 0.0,
            "anomaly_score": float(prediction),
            "is_threat": False,
            "is_anomaly": prediction > 0.8,
            "confidence": float(confidence),
        }
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input data format"
        )

@app.post("/full-analysis", response_model=DetectionResult, tags=["Detection"])
async def full_analysis(data: NetworkData):
    """Complete threat and anomaly analysis in one call"""
    threat_result = await detect_threat(data)
    anomaly_result = await detect_anomaly(data)
    
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": data.timestamp or datetime.utcnow().isoformat(),
        "threat_level": threat_result["threat_level"],
        "anomaly_score": anomaly_result["anomaly_score"],
        "is_threat": threat_result["is_threat"],
        "is_anomaly": anomaly_result["is_anomaly"],
        "confidence": (threat_result["confidence"] + anomaly_result["confidence"]) / 2,
    }

# Store startup time for uptime calculation
app.startup_time = time.time()