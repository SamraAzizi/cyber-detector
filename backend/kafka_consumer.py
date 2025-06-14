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

    def start_heartbeat(self):
        """Periodic status updates"""
        def heartbeat():
            while self.running:
                time.sleep(Config.HEARTBEAT_INTERVAL)
                status = (
                    f"Processed {self.message_count} messages | "
                    f"Active alerts: {len(self.alert_history)} | "
                    f"Last alert: {self.alert_history[-1].timestamp if self.alert_history else 'Never'}"
                )
                print(colored(f"[STATUS] {datetime.now().isoformat()} - {status}", 'blue'))
                logger.info(status)
        
        threading.Thread(target=heartbeat, daemon=True).start()

    def process_message(self, data):
        """Process a single Kafka message"""
        self.message_count += 1
        
        try:
            # Extract features and metadata
            features = data.get('features', [])
            metadata = {
                'source_ip': data.get('source_ip', 'unknown'),
                'dest_ip': data.get('destination_ip', 'unknown'),
                'protocol': data.get('protocol', 'unknown'),
                'timestamp': data.get('timestamp', datetime.now().isoformat())
            }

            # Make predictions
            threat_level = predict(self.models['threat'], features)
            anomaly_score = predict(self.models['anomaly'], features)

            # Process results
            self.evaluate_threat(threat_level, anomaly_score, metadata)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            print(colored(f"[ERROR] Message processing failed: {str(e)}", 'red'))

    def evaluate_threat(self, threat_level: float, anomaly_score: float, metadata: dict):
        """Evaluate threat levels and trigger appropriate actions"""
        timestamp = datetime.now(pytz.utc).isoformat()
        
        # Threat evaluation
        if threat_level > Config.THRESHOLDS['high_threat']:
            alert = Alert(
                type='HIGH_THREAT',
                score=threat_level,
                timestamp=timestamp,
                source_ip=metadata.get('source_ip'),
                destination_ip=metadata.get('dest_ip'),
                metadata=metadata
            )
            self.handle_alert(alert)
            
        elif threat_level > Config.THRESHOLDS['medium_threat']:
            alert = Alert(
                type='MEDIUM_THREAT',
                score=threat_level,
                timestamp=timestamp,
                source_ip=metadata.get('source_ip'),
                destination_ip=metadata.get('dest_ip'),
                metadata=metadata
            )
            self.handle_alert(alert)
            
        # Anomaly detection
        if anomaly_score > Config.THRESHOLDS['anomaly']:
            alert = Alert(
                type='ANOMALY',
                score=anomaly_score,
                timestamp=timestamp,
                source_ip=metadata.get('source_ip'),
                destination_ip=metadata.get('dest_ip'),
                metadata=metadata
            )
            self.handle_alert(alert)
