import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, model, X, y_true):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None


        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_proba),
                'pr_auc': average_precision_score(y_true, y_proba),
                'precision_recall_curve': self._get_precision_recall_curve(y_true, y_proba)
            })
            
            # Generate plots
            self._plot_roc_curve(y_true, y_proba)
            self._plot_precision_recall(y_true, y_proba)
            self._plot_confusion_matrix(y_true, y_pred)
            
        return metrics