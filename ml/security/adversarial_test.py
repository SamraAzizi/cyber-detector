from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def run_adversarial_test(model_path, X_val_path, y_val_path):
    model = joblib.load(model_path)
    X_val = pd.read_csv(X_val_path).values
    y_val = pd.read_csv(y_val_path).values

    classifier = SklearnClassifier(model=model)
    attack = FastGradientMethod(estimator=classifier, eps=0.2)

    X_adv = attack.generate(X_val)
    y_pred = classifier.predict(X_adv)
    y_pred_labels = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred

    acc = accuracy_score(y_val, y_pred_labels)
    print(f"Adversarial Accuracy: {acc:.4f}")