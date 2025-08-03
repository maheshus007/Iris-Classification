from pydantic import BaseModel, Field, confloat
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd
from fastapi import FastAPI
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from the .pkl file
model = joblib.load('../model/iris_model.pkl')

# Prometheus Instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

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

# Define a POST endpoint at "/predict" that accepts JSON input conforming to the InputData schema
@app.post("/predict")
def predict(features: InputData):
    # Convert the incoming structured input (Pydantic model) to a dictionary, 
    # then wrap it in a list to create a DataFrame with one row
    input_df = pd.DataFrame([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]], columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    
    # Use the preloaded machine learning model to make a prediction on the input data
    prediction = model.predict(input_df)
    
    # Return the prediction result as a JSON response with key "prediction"
    return {"prediction": prediction[0]}

