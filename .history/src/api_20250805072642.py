from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Annotated
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, Info
import pandas as pd
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import subprocess
import sys
import hashlib
import json
import asyncio
from pathlib import Path
import threading
import time

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

# PROMETHEUS CUSTOM METRICS
# Counters for tracking events
prediction_requests_total = Counter(
    'iris_prediction_requests_total',
    'Total number of prediction requests',
    ['method', 'endpoint', 'species']
)

training_requests_total = Counter(
    'iris_training_requests_total',
    'Total number of training requests',
    ['trigger_type']
)

data_ingestion_total = Counter(
    'iris_data_ingestion_total',
    'Total number of data samples ingested',
    ['source', 'species']
)

validation_errors_total = Counter(
    'iris_validation_errors_total',
    'Total number of validation errors',
    ['error_type', 'field']
)

# Histograms for tracking latency
prediction_duration_seconds = Histogram(
    'iris_prediction_duration_seconds',
    'Time spent on prediction requests',
    ['method', 'endpoint']
)

training_duration_seconds = Histogram(
    'iris_training_duration_seconds',
    'Time spent on model training',
    ['trigger_type']
)

# Gauges for current state
model_accuracy = Gauge(
    'iris_model_accuracy',
    'Current model accuracy'
)

dataset_size = Gauge(
    'iris_dataset_size',
    'Current size of the training dataset'
)

model_version = Info(
    'iris_model_version',
    'Current model version and metadata'
)

prediction_confidence = Histogram(
    'iris_prediction_confidence',
    'Distribution of prediction confidence scores',
    ['species']
)

data_quality_score = Gauge(
    'iris_data_quality_score',
    'Current data quality score'
)

pending_samples = Gauge(
    'iris_pending_samples',
    'Number of samples pending for retraining'
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

# =============================================================================
# DATA MONITORING AND RE-TRAINING TRIGGER SYSTEM
# =============================================================================

# Global variables for data monitoring
data_path = 'data/iris.csv'
training_data_path = 'data/training_buffer.csv'
data_change_threshold = 0.1  # 10% change in data to trigger retraining
min_new_samples = 10  # Minimum new samples before considering retraining
last_training_time = datetime.now()
auto_retrain_enabled = True
retraining_in_progress = False
data_monitoring_enabled = True

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Data monitoring state
data_monitoring_state = {
    "last_data_hash": None,
    "last_data_size": 0,
    "last_check_time": datetime.now(),
    "data_changes_detected": 0,
    "pending_samples": 0,
    "last_model_version": None,
    "retraining_history": []
}

def get_data_hash(file_path: str) -> str:
    """Calculate hash of data file for change detection."""
    try:
        if not os.path.exists(file_path):
            return ""
        
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logging.error(f"Error calculating data hash: {e}")
        return ""

def get_data_stats(file_path: str) -> Dict[str, Any]:
    """Get statistics about the data file."""
    try:
        if not os.path.exists(file_path):
            return {"size": 0, "samples": 0, "features": 0}
        
        df = pd.read_csv(file_path)
        return {
            "size": os.path.getsize(file_path),
            "samples": len(df),
            "features": len(df.columns),
            "species_distribution": df['Species'].value_counts().to_dict() if 'Species' in df.columns else {}
        }
    except Exception as e:
        logging.error(f"Error getting data stats: {e}")
        return {"size": 0, "samples": 0, "features": 0}

def initialize_data_monitoring():
    """Initialize data monitoring state."""
    global data_monitoring_state
    
    try:
        # Initialize with current data state
        current_hash = get_data_hash(data_path)
        current_stats = get_data_stats(data_path)
        
        data_monitoring_state.update({
            "last_data_hash": current_hash,
            "last_data_size": current_stats["samples"],
            "last_check_time": datetime.now(),
            "data_changes_detected": 0,
            "pending_samples": 0
        })
        
        logging.info(f"Data monitoring initialized. Current data: {current_stats['samples']} samples")
        
    except Exception as e:
        logging.error(f"Error initializing data monitoring: {e}")

# Initialize data monitoring
initialize_data_monitoring()

# INITIALIZE PROMETHEUS METRICS ON STARTUP
def initialize_prometheus_metrics():
    """Initialize baseline Prometheus metrics."""
    try:
        # Set initial dataset size
        if os.path.exists("data/iris.csv"):
            df = pd.read_csv("data/iris.csv")
            dataset_size.set(len(df))
            
            # Set initial data quality score
            data_quality_score.set(1.0)
            
        # Set initial model info if model exists
        if os.path.exists(model_path):
            model_version.info({
                'version': 'v1.0',
                'model_type': 'RandomForest',
                'status': 'loaded',
                'last_updated': datetime.now().isoformat()
            })
            # Set a default accuracy (will be updated on actual training)
            model_accuracy.set(0.95)
        
        # Initialize pending samples
        pending_samples.set(0)
        
        logging.info("Prometheus metrics initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing Prometheus metrics: {e}")

# Initialize metrics
initialize_prometheus_metrics()

# PROMETHEUS INSTRUMENTATOR SETUP
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

# Custom instrumentator functions for ML metrics
def track_prediction_metrics():
    """Custom instrumentator function to track prediction metrics."""
    def instrumentation(info):
        if info.request.url.path in ["/predict", "/predict/batch"]:
            prediction_requests_total.labels(
                method=info.request.method,
                endpoint=info.request.url.path,
                species="unknown"
            ).inc()
    return instrumentation

def track_request_duration():
    """Custom instrumentator function to track request duration."""
    def instrumentation(info):
        if info.request.url.path in ["/predict", "/predict/batch"]:
            prediction_duration_seconds.labels(
                method=info.request.method,
                endpoint=info.request.url.path
            ).observe(info.response.process_time)
        elif info.request.url.path == "/train":
            training_duration_seconds.labels(
                trigger_type="manual"
            ).observe(info.response.process_time)
    return instrumentation

# Add custom metrics to instrumentator
instrumentator.add(track_prediction_metrics())
instrumentator.add(track_request_duration())

# Instrument and expose the app
instrumentator.instrument(app)
instrumentator.expose(app)

# Manual metrics endpoint for debugging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

@app.get("/metrics")
async def get_metrics():
    """Manual Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# PYDANTIC MODELS FOR INPUT/OUTPUT VALIDATION
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
    ml_model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: Optional[str] = Field("1.0.0", description="API version")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Service uptime in seconds")

    model_config = {
        "protected_namespaces": ()
    }


# =============================================================================
# NEW DATA AND RE-TRAINING MODELS
# =============================================================================

class NewDataSample(BaseModel):
    """Schema for adding new training data samples."""
    sepal_length: float = Field(..., ge=0.1, le=15.0, description="Sepal length in centimeters")
    sepal_width: float = Field(..., ge=0.1, le=10.0, description="Sepal width in centimeters") 
    petal_length: float = Field(..., ge=0.1, le=15.0, description="Petal length in centimeters")
    petal_width: float = Field(..., ge=0.0, le=10.0, description="Petal width in centimeters")
    species: str = Field(..., description="True species label", pattern="^(Iris-setosa|Iris-versicolor|Iris-virginica)$")
    source: Optional[str] = Field("api", description="Source of the data sample")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the label")

    @validator('species')
    def validate_species(cls, v):
        """Validate species is one of the known iris types."""
        valid_species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        if v not in valid_species:
            raise ValueError(f'Species must be one of: {valid_species}')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
                "species": "Iris-setosa",
                "source": "field_collection",
                "confidence": 0.95
            }
        }
    }


class BatchNewDataRequest(BaseModel):
    """Schema for adding multiple new training data samples."""
    samples: List[NewDataSample] = Field(..., min_items=1, max_items=1000, description="List of new data samples")
    auto_trigger_retrain: bool = Field(True, description="Automatically trigger retraining if thresholds are met")
    source_description: Optional[str] = Field(None, description="Description of the data source")
    
    @validator('samples')
    def validate_samples_length(cls, v):
        if len(v) < 1:
            raise ValueError('At least one sample is required')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 samples allowed per batch')
        return v


class DataIngestionResponse(BaseModel):
    """Response schema for data ingestion operations."""
    message: str = Field(..., description="Ingestion status message")
    samples_added: int = Field(..., ge=0, description="Number of samples successfully added")
    samples_rejected: int = Field(..., ge=0, description="Number of samples rejected due to validation")
    total_dataset_size: int = Field(..., ge=0, description="Total size of dataset after ingestion")
    retraining_triggered: bool = Field(..., description="Whether automatic retraining was triggered")
    retraining_reason: Optional[str] = Field(None, description="Reason for triggering retraining")
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score of ingested data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Ingestion timestamp")


class RetrainingTriggerResponse(BaseModel):
    """Response schema for retraining trigger operations."""
    triggered: bool = Field(..., description="Whether retraining was triggered")
    reason: str = Field(..., description="Reason for decision")
    current_data_size: int = Field(..., ge=0, description="Current size of training dataset")
    new_samples_since_last_training: int = Field(..., ge=0, description="New samples added since last training")
    data_change_percentage: float = Field(..., ge=0.0, description="Percentage change in data since last training")
    estimated_training_time_minutes: Optional[float] = Field(None, ge=0, description="Estimated training time")
    scheduled_training_time: Optional[datetime] = Field(None, description="When training is scheduled to start")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class DataMonitoringStatus(BaseModel):
    """Response schema for data monitoring status."""
    monitoring_enabled: bool = Field(..., description="Whether data monitoring is active")
    auto_retrain_enabled: bool = Field(..., description="Whether automatic retraining is enabled")
    current_data_size: int = Field(..., ge=0, description="Current training dataset size")
    last_training_time: datetime = Field(..., description="Last time model was trained")
    last_data_change_time: Optional[datetime] = Field(None, description="Last time data changed")
    pending_samples: int = Field(..., ge=0, description="Number of samples pending for training")
    retraining_in_progress: bool = Field(..., description="Whether retraining is currently in progress")
    next_scheduled_check: Optional[datetime] = Field(None, description="Next scheduled data check")
    data_change_threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold for triggering retraining")
    min_samples_threshold: int = Field(..., ge=1, description="Minimum new samples needed for retraining")
    retraining_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of retraining events")


class RetrainingConfig(BaseModel):
    """Schema for configuring retraining parameters."""
    auto_retrain_enabled: bool = Field(True, description="Enable/disable automatic retraining")
    data_change_threshold: float = Field(0.1, ge=0.01, le=1.0, description="Threshold for data change (0.01-1.0)")
    min_new_samples: int = Field(10, ge=1, le=1000, description="Minimum new samples before retraining")
    max_training_interval_hours: int = Field(24, ge=1, le=168, description="Maximum hours between training sessions")
    monitoring_interval_minutes: int = Field(15, ge=1, le=1440, description="Data monitoring check interval")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum data quality score for training")


# =============================================================================
# CORE RE-TRAINING AND DATA MONITORING FUNCTIONS
# =============================================================================

def check_data_changes() -> Dict[str, Any]:
    """Check if significant data changes have occurred since last training."""
    global data_monitoring_state, last_training_time
    
    try:
        current_hash = get_data_hash(data_path)
        current_stats = get_data_stats(data_path)
        
        # Calculate change metrics
        data_changed = current_hash != data_monitoring_state["last_data_hash"]
        samples_added = current_stats["samples"] - data_monitoring_state["last_data_size"]
        change_percentage = (samples_added / max(data_monitoring_state["last_data_size"], 1)) * 100
        
        # Check time since last training
        time_since_training = datetime.now() - last_training_time
        hours_since_training = time_since_training.total_seconds() / 3600
        
        # Determine if retraining should be triggered
        should_retrain = False
        reasons = []
        
        if samples_added >= min_new_samples:
            should_retrain = True
            reasons.append(f"Added {samples_added} new samples (threshold: {min_new_samples})")
        
        if change_percentage >= (data_change_threshold * 100):
            should_retrain = True
            reasons.append(f"Data changed by {change_percentage:.1f}% (threshold: {data_change_threshold*100}%)")
        
        if hours_since_training >= 24:  # Force retrain after 24 hours
            should_retrain = True
            reasons.append(f"Last training was {hours_since_training:.1f} hours ago")
        
        return {
            "data_changed": data_changed,
            "samples_added": samples_added,
            "change_percentage": change_percentage,
            "should_retrain": should_retrain and auto_retrain_enabled,
            "reasons": reasons,
            "current_stats": current_stats,
            "hours_since_training": hours_since_training
        }
        
    except Exception as e:
        logging.error(f"Error checking data changes: {e}")
        return {
            "data_changed": False,
            "samples_added": 0,
            "change_percentage": 0.0,
            "should_retrain": False,
            "reasons": [f"Error checking data: {str(e)}"],
            "current_stats": {"samples": 0, "size": 0},
            "hours_since_training": 0
        }


def add_new_data_samples(samples: List[NewDataSample], source_desc: str = None) -> Dict[str, Any]:
    """Add new data samples to the training dataset."""
    try:
        # Load existing data or create new DataFrame
        if os.path.exists(data_path):
            existing_df = pd.read_csv(data_path)
        else:
            existing_df = pd.DataFrame(columns=['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
        
        # Convert new samples to DataFrame format
        new_data = []
        samples_added = 0
        samples_rejected = 0
        
        for i, sample in enumerate(samples):
            try:
                # Validate sample data quality
                quality_score = validate_sample_quality(sample)
                
                if quality_score >= 0.5:  # Basic quality threshold
                    new_row = {
                        'Id': len(existing_df) + len(new_data) + 1,
                        'SepalLengthCm': sample.sepal_length,
                        'SepalWidthCm': sample.sepal_width,
                        'PetalLengthCm': sample.petal_length,
                        'PetalWidthCm': sample.petal_width,
                        'Species': sample.species
                    }
                    new_data.append(new_row)
                    samples_added += 1
                else:
                    samples_rejected += 1
                    logging.warning(f"Sample rejected due to low quality score: {quality_score}")
                    
            except Exception as e:
                samples_rejected += 1
                logging.error(f"Error processing sample {i}: {e}")
        
        # Add new samples to existing data
        if new_data:
            new_df = pd.DataFrame(new_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save updated dataset
            combined_df.to_csv(data_path, index=False)
            
            # Update monitoring state
            global data_monitoring_state
            data_monitoring_state.update({
                "last_data_hash": get_data_hash(data_path),
                "last_data_size": len(combined_df),
                "last_check_time": datetime.now(),
                "pending_samples": data_monitoring_state.get("pending_samples", 0) + samples_added
            })
            
            logging.info(f"Added {samples_added} new samples to dataset. Total size: {len(combined_df)}")
        
        # Calculate overall data quality score
        overall_quality = samples_added / (samples_added + samples_rejected) if (samples_added + samples_rejected) > 0 else 0.0
        
        return {
            "samples_added": samples_added,
            "samples_rejected": samples_rejected,
            "total_dataset_size": len(existing_df) + samples_added,
            "data_quality_score": overall_quality,
            "success": samples_added > 0
        }
        
    except Exception as e:
        logging.error(f"Error adding new data samples: {e}")
        return {
            "samples_added": 0,
            "samples_rejected": len(samples),
            "total_dataset_size": 0,
            "data_quality_score": 0.0,
            "success": False,
            "error": str(e)
        }


def validate_sample_quality(sample: NewDataSample) -> float:
    """Validate the quality of a data sample and return a quality score (0-1)."""
    try:
        quality_score = 1.0
        
        # Check for reasonable measurement ranges
        measurements = [sample.sepal_length, sample.sepal_width, sample.petal_length, sample.petal_width]
        
        # Penalize extreme values
        for measurement in measurements:
            if measurement < 0.5 or measurement > 12.0:
                quality_score -= 0.2
        
        # Check for realistic ratios
        sepal_ratio = sample.sepal_length / sample.sepal_width
        if sepal_ratio < 0.5 or sepal_ratio > 5.0:
            quality_score -= 0.1
            
        petal_ratio = sample.petal_length / max(sample.petal_width, 0.1)
        if petal_ratio < 0.5 or petal_ratio > 20.0:
            quality_score -= 0.1
        
        # Bonus for confidence information
        if sample.confidence and sample.confidence > 0.8:
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception as e:
        logging.error(f"Error validating sample quality: {e}")
        return 0.5  # Default moderate quality score


async def trigger_background_retraining(reason: str = "Manual trigger"):
    """Trigger model retraining in the background."""
    global retraining_in_progress, last_training_time, model, data_monitoring_state
    
    if retraining_in_progress:
        logging.warning("Retraining already in progress, skipping trigger")
        return False
    
    try:
        retraining_in_progress = True
        start_time_retrain = time.time()
        
        logging.info(f"Starting background retraining. Reason: {reason}")
        
        # Update retraining history
        retraining_event = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "data_size_before": data_monitoring_state.get("last_data_size", 0),
            "status": "started"
        }
        
        # Perform retraining
        result = train_and_save_model()
        
        # Reload the model
        model = joblib.load(model_path)
        
        # Update timing and state
        training_time = (time.time() - start_time_retrain) * 1000
        last_training_time = datetime.now()
        
        # Update retraining history
        retraining_event.update({
            "status": "completed",
            "training_time_ms": training_time,
            "model_selected": result.get("model", "unknown"),
            "accuracy": result.get("accuracy", 0.0),
            "data_size_after": get_data_stats(data_path)["samples"]
        })
        
        # Reset pending samples counter
        data_monitoring_state["pending_samples"] = 0
        
        # Store retraining history (keep last 10 events)
        if "retraining_history" not in data_monitoring_state:
            data_monitoring_state["retraining_history"] = []
        
        data_monitoring_state["retraining_history"].append(retraining_event)
        data_monitoring_state["retraining_history"] = data_monitoring_state["retraining_history"][-10:]
        
        logging.info(f"Background retraining completed in {training_time:.2f}ms. New model: {result.get('model', 'unknown')}")
        return True
        
    except Exception as e:
        logging.error(f"Error during background retraining: {e}")
        
        # Update retraining history with error
        if 'retraining_event' in locals():
            retraining_event.update({
                "status": "failed",
                "error": str(e)
            })
            data_monitoring_state["retraining_history"].append(retraining_event)
        
        return False
        
    finally:
        retraining_in_progress = False


def evaluate_retraining_need() -> RetrainingTriggerResponse:
    """Evaluate whether retraining should be triggered and return detailed response."""
    try:
        change_info = check_data_changes()
        current_stats = change_info["current_stats"]
        
        # Estimate training time based on data size
        estimated_time_minutes = max(1.0, current_stats["samples"] / 100.0)  # Rough estimate
        
        return RetrainingTriggerResponse(
            triggered=change_info["should_retrain"],
            reason="; ".join(change_info["reasons"]) if change_info["reasons"] else "No retraining needed",
            current_data_size=current_stats["samples"],
            new_samples_since_last_training=change_info["samples_added"],
            data_change_percentage=change_info["change_percentage"],
            estimated_training_time_minutes=estimated_time_minutes,
            scheduled_training_time=datetime.now() + timedelta(minutes=1) if change_info["should_retrain"] else None
        )
        
    except Exception as e:
        logging.error(f"Error evaluating retraining need: {e}")
        return RetrainingTriggerResponse(
            triggered=False,
            reason=f"Error evaluating retraining need: {str(e)}",
            current_data_size=0,
            new_samples_since_last_training=0,
            data_change_percentage=0.0
        )

# GLOBAL VARIABLES AND INITIALIZATION
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
            ml_model_loaded=model is not None,
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
    start_time = time.time()
    predicted_species = "unknown"
    
    try:
        if model is None:
            validation_errors_total.labels(error_type="model_not_loaded", field="global").inc()
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
        predicted_species = str(prediction)
        
        # Get prediction probabilities if available
        probabilities = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_df)[0]
                classes = model.classes_ if hasattr(model, 'classes_') else ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
                confidence = float(max(proba))
                
                # Track prediction confidence
                prediction_confidence.labels(species=predicted_species).observe(confidence)
            except Exception as e:
                logging.warning(f"Could not get prediction probabilities: {e}")

        # Track prediction metrics
        prediction_requests_total.labels(
            method="POST", 
            endpoint="/predict", 
            species=predicted_species
        ).inc()
        
        # Track prediction duration
        duration = time.time() - start_time
        prediction_duration_seconds.labels(
            method="POST", 
            endpoint="/predict"
        ).observe(duration)

        # Log input and prediction
        logging.info(f"Input: {input_data} | Prediction: {prediction} | Confidence: {confidence}")

        return PredictionResponse(
            prediction=str(prediction),
            confidence=confidence,
            probabilities=probabilities,
            input_features=features,
            ml_model_version="v1.0"
        )
    except HTTPException:
        # Track validation errors for HTTP exceptions
        if predicted_species == "unknown":
            validation_errors_total.labels(error_type="http_exception", field="global").inc()
        raise
    except Exception as e:
        # Track other errors
        validation_errors_total.labels(error_type="prediction_error", field="global").inc()
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
                    ml_model_version="v1.0"
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
        # Track training request
        training_requests_total.labels(trigger_type="manual").inc()
        
        result = train_and_save_model()
        model = joblib.load(model_path)
        
        training_time = (time.time() - start_time_training) * 1000  # Convert to milliseconds
        
        # Update model metrics
        model_accuracy.set(result["accuracy"])
        model_version.info({
            'version': 'v1.0',
            'model_type': result["model"],
            'training_time': str(training_time),
            'timestamp': datetime.now().isoformat()
        })
        
        # Track training duration
        training_duration_seconds.labels(trigger_type="manual").observe(time.time() - start_time_training)
        
        # Update dataset size if available
        try:
            if os.path.exists("data/iris.csv"):
                df = pd.read_csv("data/iris.csv")
                dataset_size.set(len(df))
        except Exception as e:
            logging.warning(f"Could not update dataset size metric: {e}")
        
        return TrainingResponse(
            message="Model retrained and best model selected successfully.",
            selected_model=result["model"],
            accuracy=result["accuracy"],
            training_time_ms=training_time,
            models_tested=result.get("models_tested", 2)
        )
    except Exception as e:
        validation_errors_total.labels(error_type="training_error", field="global").inc()
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


# ADDITIONAL VALIDATION AND UTILITY ENDPOINTS
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


# =============================================================================
# NEW DATA INGESTION AND RE-TRAINING ENDPOINTS
# =============================================================================

@app.post("/data/add-sample", response_model=DataIngestionResponse)
async def add_single_data_sample(sample: NewDataSample, background_tasks: BackgroundTasks):
    """Add a single new data sample and optionally trigger retraining."""
    try:
        # Track data ingestion
        data_ingestion_total.labels(
            source=sample.source or "unknown",
            species=sample.species
        ).inc()
        
        # Add the sample to the dataset
        result = add_new_data_samples([sample], "single_sample_api")
        
        if not result["success"]:
            validation_errors_total.labels(error_type="data_ingestion_failed", field="sample").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to add sample: {result.get('error', 'Unknown error')}"
            )
        
        # Update metrics
        dataset_size.set(result["total_dataset_size"])
        data_quality_score.set(result["data_quality_score"])
        
        # Check if retraining should be triggered
        retraining_check = evaluate_retraining_need()
        retraining_triggered = False
        retraining_reason = None
        
        if retraining_check.triggered and auto_retrain_enabled:
            # Track automatic retraining trigger
            training_requests_total.labels(trigger_type="auto_new_data").inc()
            
            # Schedule background retraining
            background_tasks.add_task(trigger_background_retraining, "New data sample added")
            retraining_triggered = True
            retraining_reason = retraining_check.reason
            
        return DataIngestionResponse(
            message="Sample added successfully",
            samples_added=result["samples_added"],
            samples_rejected=result["samples_rejected"],
            total_dataset_size=result["total_dataset_size"],
            retraining_triggered=retraining_triggered,
            retraining_reason=retraining_reason,
            data_quality_score=result["data_quality_score"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error adding data sample: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add data sample: {str(e)}"
        )


@app.post("/data/add-batch", response_model=DataIngestionResponse)
async def add_batch_data_samples(request: BatchNewDataRequest, background_tasks: BackgroundTasks):
    """Add multiple new data samples and optionally trigger retraining."""
    try:
        # Add the samples to the dataset
        result = add_new_data_samples(request.samples, request.source_description)
        
        if not result["success"] and result["samples_added"] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to add any samples: {result.get('error', 'All samples rejected')}"
            )
        
        # Check if retraining should be triggered
        retraining_triggered = False
        retraining_reason = None
        
        if request.auto_trigger_retrain and auto_retrain_enabled:
            retraining_check = evaluate_retraining_need()
            
            if retraining_check.triggered:
                # Schedule background retraining
                background_tasks.add_task(trigger_background_retraining, f"Batch data ingestion: {len(request.samples)} samples")
                retraining_triggered = True
                retraining_reason = retraining_check.reason
        
        return DataIngestionResponse(
            message=f"Batch processing completed. Added {result['samples_added']} samples, rejected {result['samples_rejected']}",
            samples_added=result["samples_added"],
            samples_rejected=result["samples_rejected"],
            total_dataset_size=result["total_dataset_size"],
            retraining_triggered=retraining_triggered,
            retraining_reason=retraining_reason,
            data_quality_score=result["data_quality_score"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error adding batch data samples: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add batch data samples: {str(e)}"
        )


@app.post("/retrain/trigger", response_model=RetrainingTriggerResponse)
async def trigger_retraining(background_tasks: BackgroundTasks, force: bool = False):
    """Manually trigger model retraining or evaluate if retraining is needed."""
    try:
        # Evaluate retraining need
        retraining_evaluation = evaluate_retraining_need()
        
        # Force retraining if requested
        if force or retraining_evaluation.triggered:
            if not retraining_in_progress:
                # Schedule background retraining
                reason = "Manual trigger (forced)" if force else retraining_evaluation.reason
                background_tasks.add_task(trigger_background_retraining, reason)
                
                retraining_evaluation.triggered = True
                retraining_evaluation.reason = reason
                retraining_evaluation.scheduled_training_time = datetime.now() + timedelta(seconds=10)
            else:
                retraining_evaluation.reason = "Retraining already in progress"
                retraining_evaluation.triggered = False
        
        return retraining_evaluation
        
    except Exception as e:
        logging.error(f"Error triggering retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger retraining: {str(e)}"
        )


@app.get("/data/monitoring-status", response_model=DataMonitoringStatus)
async def get_data_monitoring_status():
    """Get the current status of data monitoring and retraining system."""
    try:
        global data_monitoring_state, last_training_time, auto_retrain_enabled, data_monitoring_enabled
        global retraining_in_progress, data_change_threshold, min_new_samples
        
        current_stats = get_data_stats(data_path)
        current_pending = data_monitoring_state.get("pending_samples", 0)
        
        # Update Prometheus metrics
        dataset_size.set(current_stats["samples"])
        pending_samples.set(current_pending)
        
        return DataMonitoringStatus(
            monitoring_enabled=data_monitoring_enabled,
            auto_retrain_enabled=auto_retrain_enabled,
            current_data_size=current_stats["samples"],
            last_training_time=last_training_time,
            last_data_change_time=data_monitoring_state.get("last_check_time"),
            pending_samples=current_pending,
            retraining_in_progress=retraining_in_progress,
            next_scheduled_check=datetime.now() + timedelta(minutes=15),  # Next monitoring check
            data_change_threshold=data_change_threshold,
            min_samples_threshold=min_new_samples,
            retraining_history=data_monitoring_state.get("retraining_history", [])
        )
        
    except Exception as e:
        logging.error(f"Error getting monitoring status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring status: {str(e)}"
        )


@app.post("/data/configure-retraining")
async def configure_retraining(config: RetrainingConfig):
    """Configure retraining parameters and data monitoring settings."""
    try:
        global auto_retrain_enabled, data_change_threshold, min_new_samples, data_monitoring_enabled
        
        # Update global configuration
        auto_retrain_enabled = config.auto_retrain_enabled
        data_change_threshold = config.data_change_threshold
        min_new_samples = config.min_new_samples
        data_monitoring_enabled = True  # Enable monitoring when configuring
        
        logging.info(f"Retraining configuration updated: auto_retrain={auto_retrain_enabled}, "
                    f"threshold={data_change_threshold}, min_samples={min_new_samples}")
        
        return {
            "message": "Retraining configuration updated successfully",
            "configuration": {
                "auto_retrain_enabled": auto_retrain_enabled,
                "data_change_threshold": data_change_threshold,
                "min_new_samples": min_new_samples,
                "data_monitoring_enabled": data_monitoring_enabled
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logging.error(f"Error configuring retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure retraining: {str(e)}"
        )


@app.get("/data/dataset-info")
async def get_dataset_info():
    """Get comprehensive information about the current training dataset."""
    try:
        if not os.path.exists(data_path):
            return {
                "message": "No training dataset found", 
                "dataset_exists": False,
                "total_samples": 0
            }
        
        df = pd.read_csv(data_path)
        
        # Calculate dataset statistics
        species_distribution = df['Species'].value_counts().to_dict() if 'Species' in df.columns else {}
        
        # Feature statistics
        feature_stats = {}
        numeric_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        for col in numeric_columns:
            if col in df.columns:
                feature_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median())
                }
        
        # Data quality metrics
        missing_values = df.isnull().sum().to_dict()
        duplicate_rows = int(df.duplicated().sum())
        
        return {
            "dataset_exists": True,
            "total_samples": len(df),
            "total_features": len(df.columns),
            "species_distribution": species_distribution,
            "feature_statistics": feature_stats,
            "data_quality": {
                "missing_values": missing_values,
                "duplicate_rows": duplicate_rows,
                "quality_score": max(0.0, 1.0 - (duplicate_rows / len(df)) - (sum(missing_values.values()) / (len(df) * len(df.columns))))
            },
            "file_info": {
                "file_size_bytes": os.path.getsize(data_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(data_path)).isoformat(),
                "file_hash": get_data_hash(data_path)
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logging.error(f"Error getting dataset info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset info: {str(e)}"
        )