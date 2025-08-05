from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Annotated
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
from fastapi import FastAPI, HTTPException, status
import numpy as np
import logging
from datetime import datetime
import os
import subprocess
import sys

from src.generate_model import train_and_save_model

def get_dvc_executable():
    """Get the path to the DVC executable."""
    # Try to find dvc executable in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment
        venv_scripts = os.path.join(sys.prefix, 'Scripts')
        dvc_exe = os.path.join(venv_scripts, 'dvc.exe')
        if os.path.exists(dvc_exe):
            return dvc_exe
    
    # Fallback to system dvc or just 'dvc'
    return 'dvc'

# Configure logging early
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from the .pkl file
try:
    model_path = 'model/iris_model.pkl'
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[],
    should_round_latency_decimals=True,
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app)
instrumentator.expose(app)

# =============================================================================
# PYDANTIC MODELS FOR INPUT/OUTPUT VALIDATION
# =============================================================================

class IrisFeatures(BaseModel):
    """Input schema for iris flower measurements with comprehensive validation."""
    sepal_length: float = Field(
        ..., 
        ge=0.1, 
        le=15.0,
        description="Sepal length in centimeters (0.1-15.0)",
        example=5.1
    )
    sepal_width: float = Field(
        ..., 
        ge=0.1, 
        le=10.0,
        description="Sepal width in centimeters (0.1-10.0)",
        example=3.5
    )
    petal_length: float = Field(
        ..., 
        ge=0.1, 
        le=15.0,
        description="Petal length in centimeters (0.1-15.0)",
        example=1.4
    )
    petal_width: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        description="Petal width in centimeters (0.0-10.0)",
        example=0.2
    )

    @validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def validate_positive_measurements(cls, v):
        """Ensure all measurements are positive and reasonable."""
        if v <= 0:
            raise ValueError('Measurements must be positive')
        if not isinstance(v, (int, float)):
            raise ValueError('Measurements must be numeric')
        return round(float(v), 2)

    @validator('petal_width')
    def validate_petal_width_range(cls, v, values):
        """Validate petal width is reasonable relative to petal length."""
        if 'petal_length' in values and values['petal_length'] > 0:
            if v > values['petal_length'] * 2:
                raise ValueError('Petal width seems unusually large compared to petal length')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    samples: conlist(IrisFeatures, min_items=1, max_items=100) = Field(
        ...,
        description="List of iris flower samples (1-100 samples)",
        example=[
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 6.0,
                "petal_width": 2.5
            }
        ]
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: str = Field(..., description="Predicted iris species")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    input_features: IrisFeatures = Field(..., description="Input features used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    model_version: Optional[str] = Field(None, description="Model version used")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Iris-setosa",
                "confidence": 0.99,
                "probabilities": {
                    "Iris-setosa": 0.99,
                    "Iris-versicolor": 0.01,
                    "Iris-virginica": 0.00
                },
                "input_features": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                "timestamp": "2025-08-05T01:30:00",
                "model_version": "v1.0"
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_samples: int = Field(..., ge=1, description="Total number of samples processed")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")


class TrainingResponse(BaseModel):
    """Response schema for training operations."""
    message: str = Field(..., description="Training status message")
    selected_model: str = Field(..., description="Best performing model selected")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy score")
    training_time_ms: Optional[float] = Field(None, ge=0, description="Training time in milliseconds")
    models_tested: Optional[int] = Field(None, ge=1, description="Number of models tested")
    timestamp: datetime = Field(default_factory=datetime.now, description="Training completion timestamp")


class DVCPipelineResponse(BaseModel):
    """Response schema for DVC pipeline operations."""
    message: str = Field(..., description="Pipeline execution status")
    pipeline_output: str = Field(..., description="DVC pipeline output")
    metrics: Optional[str] = Field(None, description="Pipeline metrics if available")
    execution_time_ms: Optional[float] = Field(None, ge=0, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")


class DVCStatusResponse(BaseModel):
    """Response schema for DVC status operations."""
    status: str = Field(..., description="DVC status (success/error)")
    output: str = Field(..., description="DVC status output")
    error: Optional[str] = Field(None, description="Error details if any")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status check timestamp")


class DVCMetricsResponse(BaseModel):
    """Response schema for DVC metrics operations."""
    status: str = Field(..., description="Metrics retrieval status")
    metrics: str = Field(..., description="DVC metrics output")
    parsed_metrics: Optional[Dict[str, Any]] = Field(None, description="Parsed metrics data")
    error: Optional[str] = Field(None, description="Error details if any")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics retrieval timestamp")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: Optional[str] = Field("1.0.0", description="API version")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Service uptime in seconds")

# =============================================================================

@app.get("/")
def read_root():
    try:
        return {"message": "Iris Classification API is running!", "model_loaded": model is not None}
    except Exception as e:
        return {"error": str(e)}

# Define a POST endpoint at "/predict" that accepts JSON input conforming to the InputData schema
@app.post("/predict")
def predict(features: InputData):
    try:
        if model is None:
            return {"error": "Model not loaded"}
        
        input_data = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]

        input_df = pd.DataFrame([input_data], columns=[
            "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"
        ])

        prediction = model.predict(input_df)[0]

        # Log input and prediction
        logging.info(f"Input: {input_data} | Prediction: {prediction}")

        return {"prediction": prediction}
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return {"error": str(e)}


@app.post("/train")
def train_model():
    global model
    try:
        result = train_and_save_model()
        model = joblib.load(model_path)
        return {
            "message": "Model retrained and best model selected successfully.",
            "selected_model": result["model"],
            "accuracy": result["accuracy"]
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/dvc/run-pipeline")
def run_dvc_pipeline():
    """Run the DVC pipeline to retrain models."""
    try:
        # Get the DVC executable path
        dvc_path = get_dvc_executable()
        
        # Run DVC pipeline
        result = subprocess.run([dvc_path, "repro"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            # Get metrics if available
            metrics_result = subprocess.run([dvc_path, "metrics", "show"], 
                                          capture_output=True, text=True, cwd=".")
            
            return {
                "message": "DVC pipeline executed successfully",
                "pipeline_output": result.stdout,
                "metrics": metrics_result.stdout if metrics_result.returncode == 0 else "No metrics available"
            }
        else:
            return {
                "error": "DVC pipeline failed",
                "output": result.stdout,
                "error_details": result.stderr
            }
    except Exception as e:
        return {"error": f"Failed to run DVC pipeline: {str(e)}"}


@app.get("/dvc/status")
def get_dvc_status():
    """Get DVC pipeline status."""
    try:
        dvc_path = get_dvc_executable()
        result = subprocess.run([dvc_path, "status"], 
                              capture_output=True, text=True, cwd=".")
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {"error": f"Failed to get DVC status: {str(e)}"}


@app.get("/dvc/metrics")
def get_dvc_metrics():
    """Get DVC metrics."""
    try:
        dvc_path = get_dvc_executable()
        result = subprocess.run([dvc_path, "metrics", "show"], 
                              capture_output=True, text=True, cwd=".")
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "metrics": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {"error": f"Failed to get DVC metrics: {str(e)}"}