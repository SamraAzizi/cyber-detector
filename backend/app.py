from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from model_loader import load_model, predict
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
from datetime import datetime
import uuid
import logging
import os
from pathlib import Path
import yaml
import json
from cachetools import TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration management
CONFIG_FILE = "config/config.yaml"

def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return {}

config = load_config()

# API Key Security
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if not config.get("auth_required", False):
        return True
    if api_key in config.get("api_keys", []):
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key"
    )

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
    version=config.get("version", "1.0.0"),
    contact={
        "name": "CyberSecurity Team",
        "email": config.get("contact_email", "security@cybershield.ai"),
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    root_path=config.get("root_path", ""),
)

# Mount static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Response caching
response_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute cache

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
    
    # Add security definitions
    if config.get("auth_required", False):
        openapi_schema["components"] = {
            "securitySchemes": {
                "APIKeyHeader": {
                    "type": "apiKey",
                    "name": API_KEY_NAME,
                    "in": "header"
                }
            }
        }
        openapi_schema["security"] = [{"APIKeyHeader": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Load models at startup - with better error handling
try:
    model_config = config.get("models", {})
    threat_model = load_model(model_config.get("threat_model", "models/threat_detection_model.pkl"))
    anomaly_model = load_model(model_config.get("anomaly_model", "models/anomaly_model.pkl"))
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    threat_model = None
    anomaly_model = None

# Enhanced data models
class NetworkData(BaseModel):
    """Network traffic data for analysis"""
    features: List[float] = Field(..., min_items=10, max_items=100, 
                                description="List of network traffic features (10-100 elements)")
    timestamp: Optional[str] = Field(None, example="2023-01-01T12:00:00Z",
                                   description="Timestamp of the network event in ISO format")
    source_ip: Optional[str] = Field(None, regex=r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
                                   description="Source IP address")
    destination_ip: Optional[str] = Field(None, regex=r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
                                        description="Destination IP address")
    protocol: Optional[str] = Field(None, description="Network protocol used")

class DetectionResult(BaseModel):
    """Result of threat/anomaly detection"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Timestamp of the analysis")
    threat_level: float = Field(..., ge=0, le=1, description="Threat probability (0-1)")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly score (0-1)")
    is_threat: bool = Field(..., description="True if threat detected")
    is_anomaly: bool = Field(..., description="True if anomaly detected")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    model_version: str = Field("1.0.0", description="Model version used")
    details: Optional[Dict] = Field(None, description="Additional detection details")

class HealthCheck(BaseModel):
    """System health status"""
    status: str = Field(..., description="Overall system status")
    models: dict = Field(..., description="Model status information")
    uptime: float = Field(..., description="System uptime in seconds")
    timestamp: str = Field(..., description="Current server timestamp")
    system_info: Optional[Dict] = Field(None, description="Additional system information")

class BatchRequest(BaseModel):
    """Batch detection request"""
    requests: List[NetworkData] = Field(..., max_items=100, description="List of network data to analyze")

class BatchResult(BaseModel):
    """Batch detection results"""
    results: List[DetectionResult]
    processing_time: float
    requests_processed: int

# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url}")
    start_time = time.time()
    
    # Add request metrics
    if not hasattr(app, "request_metrics"):
        app.request_metrics = {
            "total_requests": 0,
            "avg_response_time": 0,
            "last_request_time": None
        }
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_time = "{0:.2f}".format(process_time)
    logger.info(f"Request {request_id} completed: {formatted_time}ms")
    
    # Update metrics
    app.request_metrics["total_requests"] += 1
    app.request_metrics["avg_response_time"] = (
        app.request_metrics["avg_response_time"] * (app.request_metrics["total_requests"] - 1) + process_time
    ) / app.request_metrics["total_requests"]
    app.request_metrics["last_request_time"] = datetime.utcnow().isoformat()
    
    return response

# Custom docs endpoint for better Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1}
    )

# Simple dashboard endpoint
@app.get("/dashboard", include_in_schema=False, response_class=HTMLResponse)
async def dashboard():
    return """
    <html>
        <head>
            <title>CyberShield Dashboard</title>
        </head>
        <body>
            <h1>CyberShield API Dashboard</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
            <h2>System Metrics</h2>
            <div id="metrics"></div>
            <script>
                async function loadMetrics() {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('metrics').innerHTML = `
                        <p>Status: <strong>${data.status}</strong></p>
                        <p>Uptime: ${Math.floor(data.uptime/60)} minutes</p>
                        <p>Threat Model: ${data.models.threat_model}</p>
                        <p>Anomaly Model: ${data.models.anomaly_model}</p>
                    `;
                }
                loadMetrics();
                setInterval(loadMetrics, 5000);
            </script>
        </body>
    </html>
    """

# Health check endpoint with system info
@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    """Check system health and model status"""
    import platform
    import psutil
    
    system_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
    }
    
    return {
        "status": "OK",
        "models": {
            "threat_model": "loaded" if threat_model else "error",
            "anomaly_model": "loaded" if anomaly_model else "error"
        },
        "uptime": time.time() - app.startup_time,
        "timestamp": datetime.utcnow().isoformat(),
        "system_info": system_info
    }

# Metrics endpoint
@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Get system and API performance metrics"""
    metrics = {
        "requests": getattr(app, "request_metrics", {}),
        "timestamp": datetime.utcnow().isoformat()
    }
    return metrics

# Enhanced detection endpoints with caching and security
@app.post("/detect-threat", 
          response_model=DetectionResult, 
          tags=["Detection"],
          dependencies=[Depends(get_api_key)])
async def detect_threat(data: NetworkData, request: Request):
    """Detect cybersecurity threats in network traffic
    
    - **features**: Array of network traffic features (10-100 elements)
    - **timestamp**: Optional timestamp of the event
    - Returns: Detailed threat analysis
    """
    # Check cache first
    cache_key = f"threat_{hash(json.dumps(data.dict()))}"
    if cache_key in response_cache:
        logger.info("Returning cached threat detection result")
        return response_cache[cache_key]
    
    if not threat_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Threat detection model not available"
        )
   
    try:
        start_time = time.time()
        prediction = predict(threat_model, data.features)
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
        
        result = {
            "request_id": str(uuid.uuid4()),
            "timestamp": data.timestamp or datetime.utcnow().isoformat(),
            "threat_level": float(prediction),
            "anomaly_score": 0.0,
            "is_threat": prediction > 0.7,
            "is_anomaly": False,
            "confidence": float(confidence),
            "details": {
                "processing_time": time.time() - start_time,
                "model_version": "1.0.0",
                "source_ip": data.source_ip,
                "destination_ip": data.destination_ip
            }
        }
        
        # Cache the result
        response_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Threat detection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input data format"
        )

@app.post("/detect-anomaly", 
          response_model=DetectionResult, 
          tags=["Detection"],
          dependencies=[Depends(get_api_key)])
async def detect_anomaly(data: NetworkData):
    """Detect anomalies in network traffic patterns
    
    - **features**: Array of network traffic features (10-100 elements)
    - **timestamp**: Optional timestamp of the event
    - Returns: Detailed anomaly analysis
    """
    # Check cache first
    cache_key = f"anomaly_{hash(json.dumps(data.dict()))}"
    if cache_key in response_cache:
        logger.info("Returning cached anomaly detection result")
        return response_cache[cache_key]
    
    if not anomaly_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anomaly detection model not available"
        )
    
    try:
        start_time = time.time()
        prediction = predict(anomaly_model, data.features)
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
        
        result = {
            "request_id": str(uuid.uuid4()),
            "timestamp": data.timestamp or datetime.utcnow().isoformat(),
            "threat_level": 0.0,
            "anomaly_score": float(prediction),
            "is_threat": False,
            "is_anomaly": prediction > 0.8,
            "confidence": float(confidence),
            "details": {
                "processing_time": time.time() - start_time,
                "model_version": "1.0.0",
                "protocol": data.protocol
            }
        }
        
        # Cache the result
        response_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input data format"
        )

@app.post("/full-analysis", 
          response_model=DetectionResult, 
          tags=["Detection"],
          dependencies=[Depends(get_api_key)])
async def full_analysis(data: NetworkData):
    """Complete threat and anomaly analysis in one call"""
    cache_key = f"full_{hash(json.dumps(data.dict()))}"
    if cache_key in response_cache:
        logger.info("Returning cached full analysis result")
        return response_cache[cache_key]
    
    start_time = time.time()
    threat_result = await detect_threat(data)
    anomaly_result = await detect_anomaly(data)
    
    result = {
        "request_id": str(uuid.uuid4()),
        "timestamp": data.timestamp or datetime.utcnow().isoformat(),
        "threat_level": threat_result["threat_level"],
        "anomaly_score": anomaly_result["anomaly_score"],
        "is_threat": threat_result["is_threat"],
        "is_anomaly": anomaly_result["is_anomaly"],
        "confidence": (threat_result["confidence"] + anomaly_result["confidence"]) / 2,
        "details": {
            "processing_time": time.time() - start_time,
            "threat_details": threat_result.get("details", {}),
            "anomaly_details": anomaly_result.get("details", {})
        }
    }
    
    response_cache[cache_key] = result
    return result

# Batch processing endpoint
@app.post("/batch-analysis",
          response_model=BatchResult,
          tags=["Detection"],
          dependencies=[Depends(get_api_key)])
async def batch_analysis(batch: BatchRequest):
    """Process multiple detection requests in one call"""
    if len(batch.requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Maximum batch size is 100 requests"
        )
    
    start_time = time.time()
    results = []
    
    for request in batch.requests:
        try:
            result = await full_analysis(request)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process batch item: {str(e)}")
            results.append({
                "error": str(e),
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    return {
        "results": results,
        "processing_time": time.time() - start_time,
        "requests_processed": len(results)
    }

# Store startup time for uptime calculation
app.startup_time = time.time()

# Add shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CyberShield API")
    # Add any cleanup logic here

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, 
                host=config.get("host", "0.0.0.0"), 
                port=config.get("port", 8000),
                log_level=config.get("log_level", "info"))
