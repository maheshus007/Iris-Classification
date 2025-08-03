import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('../model/iris_model.pkl')

# Example input (features of a new sample)
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Convert the input data to a pandas DataFrame with feature names
new_sample_df = pd.DataFrame(new_sample, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# Make prediction using the loaded model
prediction = model.predict(new_sample_df)
print(f"Predicted class: {prediction[0]}")
