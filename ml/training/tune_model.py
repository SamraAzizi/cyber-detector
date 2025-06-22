import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..utils.config import load_config
from .evaluate_model import ModelEvaluator
import joblib
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelTuner:
    def __init__(self, model_type='random_forest'):
        self.config = load_config()
        self.model_type = model_type
        self.evaluator = ModelEvaluator()
        self.study = None
        
    def load_data(self):
        """Load processed training data"""
        X_train = pd.read_csv('ml/datasets/processed/X_train.csv')
        y_train = pd.read_csv('ml/datasets/processed/y_train.csv').squeeze()
        return X_train, y_train
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        X_train, y_train = self.load_data()

        

        if self.model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'class_weight': 'balanced'
            }
            model = RandomForestClassifier(**params)
            
        elif self.model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': self._calculate_scale_pos_weight(),
                'eval_metric': 'logloss'
            }
            model = XGBClassifier(**params)


        score = cross_val_score(
            model, X_train, y_train, 
            cv=3, scoring='f1_weighted', n_jobs=-1
        ).mean()
        
        return score
        
    def tune(self, n_trials=50):
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        self.study = study
        
        # Save best params
        best_params = study.best_params
        self._save_best_params(best_params)
        
        # Train and save best model
        best_model = self._train_best_model(best_params)
        return best_model, best_params
    



    def _train_best_model(self, params):
        """Train model with best parameters"""
        X_train, y_train = self.load_data()
        X_val = pd.read_csv('ml/datasets/processed/X_val.csv')
        y_val = pd.read_csv('ml/datasets/processed/y_val.csv').squeeze()
        
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1)
        elif self.model_type == 'xgboost':
            model = XGBClassifier(**params, scale_pos_weight=self._calculate_scale_pos_weight())
            
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluator.evaluate(model, X_val, y_val)
        
        # Save
        model_dir = Path(self.config['MODEL_SAVE_DIR']) / 'tuned_models'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f'{self.model_type}_tuned.joblib'
        joblib.dump(model, model_path)

        
