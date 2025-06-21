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
        


    def _get_precision_recall_curve(self, y_true, y_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }


    
    def _plot_roc_curve(self, y_true, y_proba):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig('ml/training/plots/roc_curve.png')
        plt.close()
        



    def _plot_precision_recall(self, y_true, y_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig('ml/training/plots/precision_recall.png')
        plt.close()
        



    def _plot_confusion_matrix(self, y_true, y_pred):
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('ml/training/plots/confusion_matrix.png')
        plt.close()
        



    def save_metrics(self, metrics, model_name, model_dir):
        """Save metrics to JSON file"""
        metrics_path = model_dir / f'{model_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")