import joblib
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_and_save_model():
    # Load dataset
    data = pd.read_csv('data/iris.csv')  # Adjust path for Docker volume if needed
    X = data.drop(['Id', 'Species'], axis=1)
    y = data['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=42)
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log model, params, and metrics
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model", registered_model_name="BestIrisModel")

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_model_name = name

    # Save best model locally
    joblib.dump(best_model, 'model/iris_model.pkl')
    
    # Save training metrics for DVC
    training_metrics = {
        "best_model": best_model_name,
        "training_accuracy": float(best_score),
        "models_tested": len(models),
        "test_samples": len(y_test),
        "train_samples": len(y_train)
    }
    
    with open('training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"Best model saved: {best_model_name} with accuracy: {best_score:.4f}")
    return {"model": best_model_name, "accuracy": best_score}