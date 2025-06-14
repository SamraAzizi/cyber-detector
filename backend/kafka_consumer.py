import json
import logging
from kafka import KafkaConsumer
from datetime import datetime
from termcolor import colored
from pyfiglet import Figlet
from model_loader import predict
import time
from collections import deque
import threading
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pytz

# Configuration
class Config:
    KAFKA_SERVERS = ['localhost:9092']
    KAFKA_TOPIC = 'network-traffic'
    MODEL_PATHS = {
        'threat': 'models/threat_detection_model.pkl',
        'anomaly': 'models/anomaly_model.pkl'
    }
    THRESHOLDS = {
        'high_threat': 0.8,
        'medium_threat': 0.6,
        'anomaly': 0.9
    }
    ALERT_THROTTLE_SECONDS = 300  # 5 minutes
    HEARTBEAT_INTERVAL = 60  # seconds

# Data Classes
@dataclass
class Alert:
    type: str
    score: float
    timestamp: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Initialize beautiful logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('threat_detection.log')
        ]
    )
    return logging.getLogger('kafka_consumer')

# Initialize pretty console output
f = Figlet(font='slant')
print(colored(f.renderText('CYBER SHIELD'), 'cyan'))
print(colored('='*60, 'blue'))
print(colored('Real-time Threat Detection System\n', 'yellow'))
print(colored('Listening for network traffic...\n', 'green'))

logger = setup_logging()

class ThreatDetector:
    def __init__(self):
        self.models = {}
        self.last_alert_time = {}
        self.alert_history = deque(maxlen=100)
        self.load_models()
        self.running = True
        self.message_count = 0

    def load_models(self):
        """Load ML models with error handling"""
        try:
            self.models['threat'] = load_model(Config.MODEL_PATHS['threat'])
            self.models['anomaly'] = load_model(Config.MODEL_PATHS['anomaly'])
            logger.info("Models loaded successfully")
            print(colored("[SUCCESS] ", 'green') + "Threat detection models loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            print(colored("[ERROR] ", 'red') + f"Model loading failed: {str(e)}")
            raise
