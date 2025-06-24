import json
from pathlib import Path
from termcolor import colored
from .model_adapter import ModelAdapter
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DeploymentValidator:
    """
    Comprehensive deployment validation that checks:
    - Model loading
    - Schema compatibility
    - Prediction consistency
    - Performance benchmarks
    """
    
    def __init__(self):
        self.adapter = ModelAdapter()
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }