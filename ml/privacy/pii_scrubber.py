import hashlib
import re
import pandas as pd
from typing import Dict, Any

class PIIScrubber:
    """Identify and sanitize PII in features."""
    
    def __init__(self):
        self.pii_patterns = {
            'ipv4': re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            'mac': re.compile(r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})"),
            'email': re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        }



    def scrub_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove/hash PII from training data"""
        df = df.copy()
        for col in df.select_dtypes(include='object'):
            df[col] = df[col].apply(self._scrub_text)
        return df



    def _scrub_text(self, text: str) -> str:
        """Replace PII with deterministic hashes"""
        if not isinstance(text, str):
            return text
            
        for pii_type, pattern in self.pii_patterns.items():
            text = pattern.sub(
                lambda m: f"{pii_type}_" + self._hash(m.group()),
                text
            )
        return text
    
    

    def _hash(self, value: str) -> str:
        """Deterministic hash for pseudonymization"""
        return hashlib.sha256(value.encode()).hexdigest()[:8]

# Usage in training pipeline:
# df = pd.read_csv("raw_data.csv")
# scrubber = PIIScrubber()
# clean_df = scrubber.scrub_dataset(df)
