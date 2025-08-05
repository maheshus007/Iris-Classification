from pydantic import BaseModel, Field, confloat
import joblib
# from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
from fastapi import FastAPI
import numpy as np
import logging
from datetime import datetime
import os

# from src.generate_model import train_and_save_model

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

# Prometheus Instrumentation - Temporarily disabled due to compatibility issues
# instrumentator = Instrumentator(
#     should_group_status_codes=False,
#     should_ignore_untemplated=True,
#     should_respect_env_var=True,
#     should_instrument_requests_inprogress=True,
#     excluded_handlers=[],
#     should_round_latency_decimals=True,
#     env_var_name="ENABLE_METRICS",
#     inprogress_name="inprogress",
#     inprogress_labels=True,
# )
# instrumentator.instrument(app)
# instrumentator.expose(app)

# Define a Pydantic model with input validation
class InputData(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length must be between 0 and 10")
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width must be between 0 and 10")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length must be between 0 and 10")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width must be between 0 and 10")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

@app.get("/")
def read_root():
    return {"message": "Iris Classification API is running!"}

# Define a POST endpoint at "/predict" that accepts JSON input conforming to the InputData schema
@app.post("/predict")
def predict(features: InputData):
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


# @app.post("/train")
# def train_model():
#     global model
#     try:
#         result = train_and_save_model()
#         model = joblib.load(model_path)
#         return {
#             "message": "Model retrained and best model selected successfully.",
#             "selected_model": result["model"],
#             "accuracy": result["accuracy"]
#         }
#     except Exception as e:
#         return {"error": str(e)}