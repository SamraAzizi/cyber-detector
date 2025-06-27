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
    



    def _check_decay(self, window=1000):
        """Alert if accuracy drops >5% from baseline"""
        if len(self.actuals) < window:
            return
            
        current_acc = accuracy_score(
            self.actuals[-window:],
            [round(p) for p in self.predictions[-window:]]
        )
        
        baseline = self._get_baseline_accuracy()
        decay = baseline - current_acc
        
        if decay > 0.05:
            logger.warning(f"Performance decay detected: {decay:.2%}")
            return True
        return False
    
    
    
    def _get_baseline_accuracy(self):
        """Get original validation accuracy"""
        with open('ml/models/deployment_packages/latest/metrics.json') as f:
            return json.load(f)['validation_accuracy']