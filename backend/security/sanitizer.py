import re
import numpy as np
from typing import Dict, Any
import logging

class InputSanitizer:
    """Prevent injection attacks in model inputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_regex_patterns()



    def _init_regex_patterns(self):
        """Define attack signature patterns"""
        self.patterns = {
            'sql_injection': re.compile(r"([';]+\s*--|UNION\s+SELECT)", re.I),
            'xss': re.compile(r"(<script|alert\(|onerror=)", re.I),
            'overflow': re.compile(r"\d{10,}")  # Prevent integer overflows
        }



    def sanitize_features(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize all input features.
        Throws SecurityException on malicious input.
        """
        sanitized = {}
        for k, v in raw_input.items():
            try:
                sanitized[k] = self._sanitize_value(k, v)
            except ValueError as e:
                self.logger.warning(f"Blocked malicious input in {k}: {v}")
                raise SecurityException(f"Invalid input in field {k}")
        return sanitized



    def _sanitize_value(self, field: str, value: Any) -> Any:
        """Field-specific sanitization"""
        if isinstance(value, str):
            if self._detect_attack(value):
                raise ValueError("Malicious pattern detected")
            return value[:100]  # Truncate long strings
            
        elif isinstance(value, (int, float)):
            if abs(value) > 1e6:  # Prevent numeric overflows
                raise ValueError("Numeric value out of bounds")
            return float(value)
            
        return value



    def _detect_attack(self, value: str) -> bool:
        """Check for known attack patterns"""
        return any(p.search(str(value)) for p in self.patterns.values())

class SecurityException(Exception):
    pass

# Usage in FastAPI endpoint:
# @app.post("/predict")
# async def predict(packet: Dict):
#     sanitized = InputSanitizer().sanitize_features(packet)
#     return model.predict(sanitized)