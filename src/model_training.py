import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import os

def train_model(X_train_path, y_train_path, X_test_path, y_test_path, model_type='rf'):
    """
    Trains and evaluates a model.
    """
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    print(f"Training {model_type} model...")
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'rf' or 'xgb'.")
        
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    print(f"Metrics for {model_type}: {metrics}")
    
    # Save model
    model_path = f"models/{model_type}_model_latest.pkl"
    joblib.dump(model, model_path)
    
    return model, metrics, model_path

if __name__ == "__main__":
    train_model('data/X_train.csv', 'data/y_train.csv', 'data/X_test.csv', 'data/y_test.csv', 'rf')
