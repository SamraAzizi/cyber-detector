import re
import hashlib

def sanitize_features(features: dict) -> dict:
    """Sanitize input features to prevent injection attacks"""
    clean = {}
    for key, value in features.items():
        if isinstance(value, str):
            value = re.sub(r"[^\w\s\-.:]", "", value)  # remove special chars
        clean[key] = value
    return clean

def hash_ip(ip_address: str) -> str:
    """Hash IP address to anonymize it"""
    return hashlib.sha256(ip_address.encode()).hexdigest()
