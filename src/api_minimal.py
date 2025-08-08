from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
import os

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
model_path = 'model/iris_model.pkl'
model = joblib.load(model_path)

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
