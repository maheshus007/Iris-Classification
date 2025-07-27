from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from the .pkl file
model = joblib.load('model/iris_model.pkl')

# Define a Pydantic model for input data validation
class InputData(BaseModel):
    features: list  # The input features will be passed as a list

@app.post("/predict")
def predict(input_data: InputData):
    # Convert the input data to a pandas DataFrame with feature names
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    features = np.array(input_data.features).reshape(1, -1)
    input_df = pd.DataFrame(features, columns=feature_names)

    # Make prediction using the loaded model
    prediction = model.predict(input_df)
    
    # Return the prediction as a JSON response
    return {"prediction": prediction[0]}
