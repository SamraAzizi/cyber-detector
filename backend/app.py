from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from model_loader import ModelLoader  # Using your developer's class
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import time
from datetime import datetime
import uuid
import logging
import os
from pathlib import Path
import yaml
import json
from cachetools import TTLCache
import platform
import psutil
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================================
# Enhanced Configuration Management
# =========================================================================
class AppConfig:
    """Centralized configuration management with validation"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_config()
        return cls._instance
        
    def load_config(self):
        """Load and validate configuration"""
        try:
            with open("config/config.yaml") as f:
                self.config = yaml.safe_load(f) or {}
            self._set_defaults()
            self._validate()
        except Exception as e:
            logger.critical(f"Config load failed: {str(e)}")
            self.config = {}
            raise
            
    def _set_defaults(self):
        """Set safe defaults for missing config values"""
        defaults = {
            'version': '1.0.0',
            'auth_required': False,
            'threat_threshold': 0.7,
            'anomaly_threshold': 0.8,
            'cache_ttl': 300,
            'model_refresh_interval': 3600,
            'cors_origins': ["*"]
        }
        for k, v in defaults.items():
            self.config.setdefault(k, v)
            
    def _validate(self):
        """Validate critical configuration"""
        if self.config.get('auth_required') and not self.config.get('api_keys'):
            logger.warning("Authentication enabled but no API keys configured")

# Initialize configuration early
try:
    app_config = AppConfig()
    config = app_config.config
except Exception as e:
    logger.critical(f"Failed to initialize configuration: {str(e)}")
    raise

# =========================================================================
# Model Management Setup
# =========================================================================
DEPLOYMENT_PACKAGE = "ml/models/deployment_packages/latest"
model_adapter = ModelLoader()

# =========================================================================
# FastAPI Application Setup
# =========================================================================
app = FastAPI(
    title="CyberShield API",
    description="""**Advanced Threat Detection & Anomaly Modeling System** 🔍🛡️
    
    This API provides real-time cybersecurity threat detection using machine learning models.
    """,
    version=config['version'],
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
response_cache = TTLCache(maxsize=1000, ttl=config['cache_ttl'])

# =========================================================================
# Model Lifecycle Management
# =========================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize application state and load models"""
    app.startup_time = time.time()
    app.request_metrics = {
        "total_requests": 0,
        "avg_response_time": 0,
        "last_request_time": None
    }
    
    try:
        if not model_adapter.load_deployment(DEPLOYMENT_PACKAGE):
            logger.error("Model deployment package failed to load")
            raise RuntimeError("Model loading failed")
        logger.info("✅ Models loaded successfully from deployment package")
        
        # Initial health check
        health = model_adapter.get_stats()
        logger.info(f"Model Health: {json.dumps(health, indent=2)}")
    except Exception as e:
        logger.critical(f"🛑 Critical model loading failure: {str(e)}")
        if config.get("strict_mode", False):
            raise

# =========================================================================
# Enhanced Data Models
# =========================================================================
class ModelHealth(BaseModel):
    """Detailed model health information"""
    name: str
    status: Literal['loading', 'healthy', 'degraded', 'error']
    version: str
    predictions: int = 0
    avg_latency: float = 0
    last_used: Optional[str] = None

class DetectionResult(BaseModel):
    """Enhanced detection result with model metadata"""
    request_id: str
    timestamp: str
    threat_level: float = Field(..., ge=0, le=1)
    anomaly_score: float = Field(..., ge=0, le=1)
    is_threat: bool
    is_anomaly: bool
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    model_name: Optional[str] = None
    details: Dict = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

# (Keep your existing NetworkData, BatchRequest, BatchResult models)

# =========================================================================
# API Endpoints
# =========================================================================
@app.get("/model-health", response_model=List[ModelHealth])
async def get_model_health():
    """Get detailed health status of all loaded models"""
    stats = model_adapter.get_stats()
    return [{
        "name": name,
        "status": "healthy" if data.get('model') else "error",
        "version": data.get('version', 'unknown'),
        "predictions": data.get('predictions', 0),
        "avg_latency": data.get('avg_pred_time', 0),
        "last_used": data.get('last_used')
    } for name, data in stats.get('models', {}).items()]

@app.post("/refresh-models", dependencies=[Depends(get_api_key)])
async def refresh_models():
    """Hot-reload models without restarting service"""
    try:
        if model_adapter.load_deployment(DEPLOYMENT_PACKAGE):
            logger.info("♻️ Models reloaded successfully")
            return {"status": "success", "message": "Models reloaded"}
        logger.error("❌ Model reload failed")
        return {"status": "error", "message": "Model reload failed"}
    except Exception as e:
        logger.error(f"Model refresh failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# Enhanced Detection Endpoints
# =========================================================================
@app.post("/detect-threat", response_model=DetectionResult)
async def detect_threat(data: NetworkData):
    """Enhanced threat detection with model adapter"""
    cache_key = f"threat_{hash(json.dumps(data.dict()))}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        # Convert to model input format
        input_features = {
            'features': data.features,
            **{k:v for k,v in data.dict().items() 
               if k in ['timestamp', 'source_ip', 'destination_ip', 'protocol']}
        }
        
        # Use adapter for prediction
        start_time = time.time()
        result = model_adapter.predict(input_features)
        processing_time = time.time() - start_time
        
        # Build response
        response = DetectionResult(
            request_id=str(uuid.uuid4()),
            timestamp=data.timestamp or datetime.utcnow().isoformat(),
            threat_level=result['prediction'],
            anomaly_score=0.0,
            is_threat=result['prediction'] > config['threat_threshold'],
            is_anomaly=False,
            confidence=result['confidence'],
            model_version=result['model_version'],
            details={
                "processing_time": processing_time,
                "source_ip": data.source_ip,
                "destination_ip": data.destination_ip,
                **({'visualization': result['visualization']} 
                   if 'visualization' in result else {})
            }
        )
        
        response_cache[cache_key] = response
        return response
        
    except Exception as e:
        logger.error(f"Threat prediction failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

# (Similarly enhance your other endpoints: detect-anomaly, full-analysis, batch-analysis)

# =========================================================================
# Existing Middleware and Supporting Functions
# =========================================================================
# (Keep your existing middleware, CORS, OpenAPI customization, etc.)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, 
                host=config.get("host", "0.0.0.0"), 
                port=config.get("port", 8000),
                log_level=config.get("log_level", "info"))