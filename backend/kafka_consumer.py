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

        # Log all predictions for debugging
        debug_msg = (
            f"Threat: {threat_level:.3f} | "
            f"Anomaly: {anomaly_score:.3f} | "
            f"From: {metadata.get('source_ip')} | "
            f"To: {metadata.get('dest_ip')} | "
            f"Protocol: {metadata.get('protocol')}"
        )
        logger.debug(debug_msg)

    def handle_alert(self, alert: Alert):
        """Process and display alerts with throttling"""
        # Throttle alerts of the same type
        last_alert = self.last_alert_time.get(alert.type, 0)
        current_time = time.time()
        
        if current_time - last_alert < Config.ALERT_THROTTLE_SECONDS:
            return
            
        self.last_alert_time[alert.type] = current_time
        self.alert_history.append(alert)
        
        # Format alert message
        if alert.type == 'HIGH_THREAT':
            alert_msg = colored(f"🚨 CRITICAL THREAT DETECTED: {alert.score:.3f}", 'red', attrs=['bold'])
        elif alert.type == 'MEDIUM_THREAT':
            alert_msg = colored(f"⚠️ POTENTIAL THREAT DETECTED: {alert.score:.3f}", 'yellow')
        else:
            alert_msg = colored(f"🔍 NETWORK ANOMALY DETECTED: {alert.score:.3f}", 'magenta')
        
        details = (
            f"\n  Timestamp: {alert.timestamp}"
            f"\n  Source IP: {alert.source_ip}"
            f"\n  Dest IP: {alert.destination_ip}"
        )
        
        # Print to console and log
        print(f"\n{alert_msg}{details}\n")
        logger.warning(f"{alert.type} - Score: {alert.score:.3f} - {details}")
        
        # Additional alert actions could go here (email, SMS, etc.)
        # self.send_email_alert(alert)

    def send_email_alert(self, alert: Alert):
        """Example email alerting function"""
        try:
            msg = MIMEText(
                f"Security Alert: {alert.type}\n"
                f"Score: {alert.score:.3f}\n"
                f"Timestamp: {alert.timestamp}\n"
                f"Source IP: {alert.source_ip}\n"
                f"Destination IP: {alert.destination_ip}"
            )
            
            msg['Subject'] = f"Security Alert: {alert.type}"
            msg['From'] = "threat-detector@yourdomain.com"
            msg['To'] = "security-team@yourdomain.com"
            
            # This would need proper SMTP configuration
            # with smtplib.SMTP('smtp.yourdomain.com') as server:
            #     server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def run(self):
        """Main consumer loop"""
        self.start_heartbeat()
        
        try:
            consumer = KafkaConsumer(
                Config.KAFKA_TOPIC,
                bootstrap_servers=Config.KAFKA_SERVERS,
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            print(colored("\n✔️ Successfully connected to Kafka\n", 'green'))
            logger.info(f"Started consuming from {Config.KAFKA_TOPIC}")
            
            for message in consumer:
                if not self.running:
                    break
                    
                self.process_message(message.value)
                
        except KeyboardInterrupt:
            print(colored("\n🛑 Gracefully shutting down...", 'yellow'))
            logger.info("Shutdown initiated by user")
        except Exception as e:
            logger.critical(f"Fatal error: {str(e)}")
            print(colored(f"\n💀 FATAL ERROR: {str(e)}", 'red'))
        finally:
            self.running = False
            summary = (
                f"\nSession Summary:\n"
                f"Messages processed: {self.message_count}\n"
                f"Alerts triggered: {len(self.alert_history)}\n"
                f"Last alert: {self.alert_history[-1].timestamp if self.alert_history else 'None'}\n"
            )
            print(colored(summary, 'cyan'))
            logger.info(summary)

if __name__ == "__main__":
    detector = ThreatDetector()
    detector.run()
