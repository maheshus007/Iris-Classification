#!/usr/bin/env python3
"""
Model evaluation script for DVC pipeline
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import os

def evaluate_model():
    """Evaluate the trained model and save metrics."""
    
    # Load data and model
    data = pd.read_csv('data/iris.csv')
    X = data.drop(['Id', 'Species'], axis=1)
    y = data['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained model
    model = joblib.load('model/iris_model.pkl')
    
    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Generate classification report
    report = classification_report(y_test, predictions, output_dict=True)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'test_samples': len(y_test),
        'precision_macro': float(report['macro avg']['precision']),
        'recall_macro': float(report['macro avg']['recall']),
        'f1_macro': float(report['macro avg']['f1-score'])
    }
    
    # Ensure metrics file exists
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'Model evaluation completed:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Test samples: {len(y_test)}')
    print(f'Precision (macro): {metrics["precision_macro"]:.4f}')
    print(f'Recall (macro): {metrics["recall_macro"]:.4f}')
    print(f'F1-score (macro): {metrics["f1_macro"]:.4f}')
    
    return metrics

if __name__ == "__main__":
    evaluate_model()
