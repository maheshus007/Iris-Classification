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

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Iris Classification API",
    description="A comprehensive ML API for iris flower classification with MLflow and DVC integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return ErrorResponse(
        error=exc.detail,
        error_code=str(exc.status_code),
        details={"status_code": exc.status_code, "path": str(request.url)}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return ErrorResponse(
        error="Validation error",
        error_code="VALIDATION_ERROR",
        details={"message": str(exc), "path": str(request.url)}
    )

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
    samples: List[IrisFeatures] = Field(
        ...,
        min_items=1,
        max_items=100,
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

    @validator('samples')
    def validate_samples_length(cls, v):
        """Validate the number of samples."""
        if len(v) < 1:
            raise ValueError('At least one sample is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 samples allowed per batch')
        return v


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: str = Field(..., description="Predicted iris species")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    input_features: IrisFeatures = Field(..., description="Input features used for prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    ml_model_version: Optional[str] = Field(None, description="Model version used")

    model_config = {
        "protected_namespaces": (),
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
                "ml_model_version": "v1.0"
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

# =============================================================================
# GLOBAL VARIABLES AND INITIALIZATION
# =============================================================================
import time
start_time = time.time()

@app.get("/", response_model=HealthResponse)
async def read_root():
    """Health check endpoint with comprehensive status information."""
    try:
        uptime = time.time() - start_time
        return HealthResponse(
            status="healthy",
            message="Iris Classification API is running!",
            model_loaded=model is not None,
            uptime_seconds=uptime
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Make a prediction for a single iris flower sample."""
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train a model first."
            )
        
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
        
        # Get prediction probabilities if available
        probabilities = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_df)[0]
                classes = model.classes_ if hasattr(model, 'classes_') else ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
                confidence = float(max(proba))
            except Exception as e:
                logging.warning(f"Could not get prediction probabilities: {e}")

        # Log input and prediction
        logging.info(f"Input: {input_data} | Prediction: {prediction} | Confidence: {confidence}")

        return PredictionResponse(
            prediction=str(prediction),
            confidence=confidence,
            probabilities=probabilities,
            input_features=features,
            model_version="v1.0"
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple iris flower samples."""
    start_time_batch = time.time()
    
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please train a model first."
            )
        
        predictions = []
        for features in request.samples:
            try:
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
                
                # Get prediction probabilities if available
                probabilities = None
                confidence = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(input_df)[0]
                        classes = model.classes_ if hasattr(model, 'classes_') else ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
                        confidence = float(max(proba))
                    except Exception as e:
                        logging.warning(f"Could not get prediction probabilities: {e}")

                predictions.append(PredictionResponse(
                    prediction=str(prediction),
                    confidence=confidence,
                    probabilities=probabilities,
                    input_features=features,
                    model_version="v1.0"
                ))
                
                # Log prediction
                logging.info(f"Batch prediction - Input: {input_data} | Prediction: {prediction}")
                
            except Exception as e:
                # Log error but continue with other samples
                logging.error(f"Error predicting sample {features}: {e}")
                continue
        
        processing_time = (time.time() - start_time_batch) * 1000  # Convert to milliseconds
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/train", response_model=TrainingResponse)
async def train_model():
    """Retrain the model and select the best performing one."""
    global model
    start_time_training = time.time()
    
    try:
        result = train_and_save_model()
        model = joblib.load(model_path)
        
        training_time = (time.time() - start_time_training) * 1000  # Convert to milliseconds
        
        return TrainingResponse(
            message="Model retrained and best model selected successfully.",
            selected_model=result["model"],
            accuracy=result["accuracy"],
            training_time_ms=training_time,
            models_tested=result.get("models_tested", 2)
        )
    except Exception as e:
        logging.error(f"Error in training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@app.post("/dvc/run-pipeline", response_model=DVCPipelineResponse)
async def run_dvc_pipeline():
    """Run the DVC pipeline to retrain models."""
    start_time_pipeline = time.time()
    
    try:
        # Get the DVC executable path
        dvc_path = get_dvc_executable()
        
        # Run DVC pipeline
        result = subprocess.run([dvc_path, "repro"], 
                              capture_output=True, text=True, cwd=".")
        
        execution_time = (time.time() - start_time_pipeline) * 1000  # Convert to milliseconds
        
        if result.returncode == 0:
            # Get metrics if available
            metrics_result = subprocess.run([dvc_path, "metrics", "show"], 
                                          capture_output=True, text=True, cwd=".")
            
            return DVCPipelineResponse(
                message="DVC pipeline executed successfully",
                pipeline_output=result.stdout,
                metrics=metrics_result.stdout if metrics_result.returncode == 0 else "No metrics available",
                execution_time_ms=execution_time
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"DVC pipeline failed: {result.stderr}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error running DVC pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run DVC pipeline: {str(e)}"
        )


@app.get("/dvc/status", response_model=DVCStatusResponse)
async def get_dvc_status():
    """Get DVC pipeline status."""
    try:
        dvc_path = get_dvc_executable()
        result = subprocess.run([dvc_path, "status"], 
                              capture_output=True, text=True, cwd=".")
        
        return DVCStatusResponse(
            status="success" if result.returncode == 0 else "error",
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else None
        )
    except Exception as e:
        logging.error(f"Error getting DVC status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DVC status: {str(e)}"
        )


@app.get("/dvc/metrics", response_model=DVCMetricsResponse)
async def get_dvc_metrics():
    """Get DVC metrics with optional parsing."""
    try:
        dvc_path = get_dvc_executable()
        result = subprocess.run([dvc_path, "metrics", "show"], 
                              capture_output=True, text=True, cwd=".")
        
        # Try to parse metrics if available
        parsed_metrics = None
        if result.returncode == 0 and result.stdout:
            try:
                # Attempt to parse metrics (this would depend on your metrics format)
                import json
                import re
                
                # Simple parsing for common metrics patterns
                metrics_lines = result.stdout.strip().split('\n')
                parsed_metrics = {}
                
                for line in metrics_lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                # Try to extract numeric values
                                for i, part in enumerate(parts[1:], 1):
                                    try:
                                        value = float(part)
                                        key = f"metric_{i}" if i == 1 else f"metric_{i}"
                                        parsed_metrics[key] = value
                                    except ValueError:
                                        continue
                            except Exception:
                                continue
                                
            except Exception as parse_error:
                logging.warning(f"Could not parse metrics: {parse_error}")
        
        return DVCMetricsResponse(
            status="success" if result.returncode == 0 else "error",
            metrics=result.stdout,
            parsed_metrics=parsed_metrics,
            error=result.stderr if result.returncode != 0 else None
        )
    except Exception as e:
        logging.error(f"Error getting DVC metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get DVC metrics: {str(e)}"
        )


# =============================================================================
# ADDITIONAL VALIDATION AND UTILITY ENDPOINTS
# =============================================================================

@app.post("/validate", response_model=Dict[str, Any])
async def validate_input(features: IrisFeatures):
    """Validate input features without making a prediction."""
    try:
        # Perform additional business logic validation
        warnings = []
        
        # Check for unusual feature combinations
        if features.petal_length < 1.0 and features.petal_width > 0.5:
            warnings.append("Unusual petal dimensions: very short length with wide width")
        
        if features.sepal_length < 4.0:
            warnings.append("Sepal length is unusually small for iris flowers")
        
        if features.sepal_width > 4.5:
            warnings.append("Sepal width is unusually large for iris flowers")
        
        # Calculate feature ratios for additional validation
        sepal_ratio = features.sepal_length / features.sepal_width
        petal_ratio = features.petal_length / features.petal_width if features.petal_width > 0 else 0
        
        return {
            "status": "valid",
            "input_features": features.dict(),
            "warnings": warnings,
            "feature_analysis": {
                "sepal_length_width_ratio": round(sepal_ratio, 2),
                "petal_length_width_ratio": round(petal_ratio, 2) if petal_ratio > 0 else None,
                "total_sepal_size": round(features.sepal_length + features.sepal_width, 2),
                "total_petal_size": round(features.petal_length + features.petal_width, 2)
            },
            "validation_timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the currently loaded model."""
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model is currently loaded"
            )
        
        model_info = {
            "model_type": type(model).__name__,
            "model_loaded": True,
            "sklearn_version": None,
            "feature_names": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            "target_classes": None,
            "model_parameters": {},
            "capabilities": {
                "predict": hasattr(model, 'predict'),
                "predict_proba": hasattr(model, 'predict_proba'),
                "feature_importance": hasattr(model, 'feature_importances_')
            }
        }
        
        # Get sklearn version if available
        try:
            import sklearn
            model_info["sklearn_version"] = sklearn.__version__
        except ImportError:
            pass
        
        # Get classes if available
        if hasattr(model, 'classes_'):
            model_info["target_classes"] = [str(cls) for cls in model.classes_]
        
        # Get model parameters if available
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                # Only include serializable parameters
                model_info["model_parameters"] = {
                    k: v for k, v in params.items() 
                    if isinstance(v, (int, float, str, bool, type(None)))
                }
            except Exception:
                pass
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            try:
                importance = model.feature_importances_.tolist()
                model_info["feature_importance"] = dict(zip(
                    model_info["feature_names"], 
                    [round(imp, 4) for imp in importance]
                ))
            except Exception:
                pass
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.get("/metrics/api")
async def get_api_metrics():
    """Get API performance metrics."""
    try:
        uptime = time.time() - start_time
        
        # You could extend this with actual metrics from a metrics store
        return {
            "uptime_seconds": uptime,
            "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "model_loaded": model is not None,
            "endpoints": {
                "total": 10,
                "prediction_endpoints": 2,
                "training_endpoints": 1,
                "dvc_endpoints": 3,
                "utility_endpoints": 4
            },
            "version": "1.0.0",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API metrics: {str(e)}"
        )