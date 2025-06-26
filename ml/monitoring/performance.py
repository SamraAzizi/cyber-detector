import pandas as pd
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:

    def __init__(self):
        self.predictions = []
        self.actuals = []
    


    def add_prediction(self, y_pred: float, y_true: float = None):
        """Log prediction with optional ground truth"""
        self.predictions.append(y_pred)
        if y_true is not None:
            self.actuals.append(y_true)
            self._check_decay()
    