import pickle
import numpy as np
import time
from datetime import datetime
from termcolor import colored
import logging
from typing import Optional, Tuple, Any
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
from sklearn.metrics import classification_report
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelLoader:
    """
    Advanced Model Loader with Performance Tracking and Visualization
    
    Features:
    - Beautiful colored console output
    - Detailed logging
    - Prediction performance metrics
    - Model validation
    - Prediction visualization
    - Memory optimization
    """
    
    def __init__(self):
        self.loaded_models = {}
        self.prediction_stats = {}
        self.logger = self._setup_logging()
        self._print_welcome()
        
    def _print_welcome(self):
        """Display beautiful welcome message"""
        print(colored("\n" + "="*60, 'blue'))
        print(colored("✨ AI MODEL LOADER v2.0", 'cyan', attrs=['bold']))
        print(colored("Advanced Threat Detection Engine", 'yellow'))
        print(colored("="*60 + "\n", 'blue'))
        
    def _setup_logging(self):
        """Configure advanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_loader.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('ModelLoader')
    
    def load_model(self, model_path: str, model_name: str = None) -> Optional[Any]:
        """
        Load a trained ML model from file with enhanced features
        
        Args:
            model_path: Path to the model file
            model_name: Friendly name for the model (optional)
            
        Returns:
            Loaded model object or None if failed
        """
        model_name = model_name or model_path.split('/')[-1]
        
        try:
            start_time = time.time()
            
            # Memory-efficient loading for large models
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            load_time = time.time() - start_time
            self.loaded_models[model_name] = {
                'model': model,
                'load_time': load_time,
                'path': model_path,
                'predictions': 0
            }
            
            # Print success message
            msg = (f"✅ Successfully loaded model '{model_name}' "
                  f"({load_time:.2f}s) | Features: {self._get_model_features(model)}")
            print(colored(msg, 'green'))
            self.logger.info(msg)
            
            return model
            
        except Exception as e:
            error_msg = f"❌ Failed to load model '{model_name}': {str(e)}"
            print(colored(error_msg, 'red'))
            self.logger.error(error_msg)
            return None
    
    def _get_model_features(self, model) -> str:
        """Get model features information if available"""
        try:
            if hasattr(model, 'n_features_in_'):
                return f"{model.n_features_in_} features"
            if hasattr(model, 'coef_'):
                return f"{len(model.coef_[0])} features"
            return "unknown features"
        except:
            return "unknown features"
    
    def predict(self, model: Any, features: list, 
                feature_names: list = None, 
                visualize: bool = False) -> Tuple[float, Optional[str]]:
        """
        Make prediction using loaded model with advanced features
        
        Args:
            model: Loaded model object
            features: Input features for prediction
            feature_names: Names of features for visualization
            visualize: Whether to generate visualization
            
        Returns:
            Tuple of (prediction, visualization_base64_or_None)
        """
        if model is None:
            error_msg = "Model not loaded - prediction aborted"
            print(colored(error_msg, 'red'))
            self.logger.error(error_msg)
            return -1, None
            
        try:
            start_time = time.time()
            features_array = np.array(features).reshape(1, -1)
            
            # Get prediction and probability if available
            prediction = model.predict(features_array)[0]
            proba = None
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_array)[0]
                confidence = max(proba)
            else:
                confidence = 1.0
                
            pred_time = time.time() - start_time
            
            # Update statistics
            model_name = self._get_model_name(model)
            if model_name in self.loaded_models:
                self.loaded_models[model_name]['predictions'] += 1
                if 'total_pred_time' not in self.loaded_models[model_name]:
                    self.loaded_models[model_name]['total_pred_time'] = 0
                self.loaded_models[model_name]['total_pred_time'] += pred_time
            
            # Generate visualization if requested
            vis_html = None
            if visualize:
                vis_html = self._generate_visualization(
                    model, features, prediction, proba, feature_names)
            
            # Log prediction
            pred_msg = (f"📊 Prediction: {prediction:.4f} | "
                       f"Confidence: {confidence:.2%} | "
                       f"Time: {pred_time:.4f}s")
            print(colored(pred_msg, 'blue'))
            self.logger.info(pred_msg)
            
            return prediction, vis_html
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(colored(error_msg, 'red'))
            self.logger.error(error_msg)
            return -1, None
    
    def _get_model_name(self, model) -> str:
        """Get name of loaded model"""
        for name, data in self.loaded_models.items():
            if data['model'] == model:
                return name
        return "unknown_model"
    
    def _generate_visualization(self, model, features, prediction, 
                              proba, feature_names) -> str:
        """
        Generate beautiful visualization of the prediction
        
        Returns:
            Base64 encoded HTML with visualization
        """
        try:
            plt.style.use('ggplot')
            fig, ax = plt.subplots(1, 2 if proba is not None else 1, 
                                 figsize=(12, 5))
            
            # Feature importance plot
            if len(features) < 30:  # Only plot if reasonable number of features
                if feature_names is None:
                    feature_names = [f'Feature {i}' for i in range(len(features))]
                
                if hasattr(model, 'coef_'):
                    importances = model.coef_[0]
                elif hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    importances = features  # Fallback to raw features
                
                pd.Series(importances, index=feature_names).plot(
                    kind='barh', ax=ax[0] if proba is not None else ax,
                    title='Feature Importance')
            
            # Probability plot if available
            if proba is not None:
                pd.Series(proba).plot(
                    kind='bar', ax=ax[1],
                    title=f'Class Probabilities (Predicted: {prediction:.2f})',
                    color=['skyblue' if i != np.argmax(proba) else 'salmon' 
                          for i in range(len(proba))])
            
            plt.tight_layout()
            
            # Save to base64 HTML
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return (f'<img src="data:image/png;base64,{img_base64}" '
                    'style="max-width:100%; height:auto;">')
            
        except Exception as e:
            self.logger.warning(f"Visualization failed: {str(e)}")
            return None
    
    def validate_model(self, model, X_test, y_test) -> Optional[str]:
        """
        Validate model performance and generate report
        
        Args:
            model: Loaded model object
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Classification report as HTML
        """
        try:
            start_time = time.time()
            y_pred = model.predict(X_test)
            val_time = time.time() - start_time
            
            report = classification_report(y_test, y_pred, output_dict=True)
            df = pd.DataFrame(report).transpose()
            
            # Print summary
            accuracy = report['accuracy']
            msg = (f"🧪 Model validation completed | "
                  f"Accuracy: {accuracy:.2%} | "
                  f"Time: {val_time:.2f}s")
            print(colored(msg, 'magenta'))
            self.logger.info(msg)
            
            # Return styled HTML table
            return (df.style
                    .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                    .format({'precision': '{:.2%}', 'recall': '{:.2%}', 
                            'f1-score': '{:.2%}', 'accuracy': '{:.2%}'})
                    .set_caption(f"Model Validation Report | Accuracy: {accuracy:.2%}")
                    .render())

            
        except Exception as e:
            error_msg = f"Model validation failed: {str(e)}"
            print(colored(error_msg, 'red'))
            self.logger.error(error_msg)
            return None
    
    def get_stats(self) -> dict:
        """Get statistics about loaded models and predictions"""
        stats = {
            'total_models': len(self.loaded_models),
            'total_predictions': sum(
                m['predictions'] for m in self.loaded_models.values()),
            'models': {}
        }
        
        for name, data in self.loaded_models.items():
            avg_time = (data['total_pred_time'] / data['predictions'] 
                       if data['predictions'] > 0 else 0)
            stats['models'][name] = {
                'load_time': data['load_time'],
                'predictions': data['predictions'],
                'avg_pred_time': avg_time,
                'path': data['path']
            }
        
        return stats
    
    def print_stats(self):
        """Display beautiful statistics summary"""
        stats = self.get_stats()
        
        print(colored("\n📈 MODEL LOADER STATISTICS", 'cyan', attrs=['bold']))
        print(colored("="*40, 'blue'))
        print(colored(f"Total Models Loaded: {stats['total_models']}", 'yellow'))
        print(colored(f"Total Predictions Made: {stats['total_predictions']}", 'yellow'))
        print(colored("-"*40, 'blue'))
        
        for name, data in stats['models'].items():
            print(colored(f"\n🔹 Model: {name}", 'green'))
            print(f"  Load Time: {data['load_time']:.2f}s")
            print(f"  Predictions: {data['predictions']}")
            print(f"  Avg. Prediction Time: {data['avg_pred_time']:.4f}s")
            print(f"  Path: {data['path']}")
        
        print(colored("\n" + "="*40 + "\n", 'blue'))

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced model loader
    loader = ModelLoader()
    
    # Load a model with beautiful feedback
    model = loader.load_model("models/threat_detection_model.pkl", "ThreatDetector")
    
    # Make a sample prediction with visualization
    if model:
        sample_features = [1.2, 0.5, 3.4, 2.1, 0.9]
        prediction, visualization = loader.predict(
            model, sample_features, 
            feature_names=["Duration", "Protocol", "SrcBytes", "DstBytes", "Count"],
            visualize=True)
        
        # Print stats
        loader.print_stats()
        
        # Example of saving visualization
        if visualization:
            with open("prediction_visualization.html", "w") as f:
                f.write(visualization)
