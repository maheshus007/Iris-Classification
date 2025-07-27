from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from the .pkl file
model = joblib.load('model/iris_model.pkl')

# Define a Pydantic model for the input data with feature names
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

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

