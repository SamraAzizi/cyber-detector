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

