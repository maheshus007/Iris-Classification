import joblib
import pandas as pd
import numpy as np
import argparse
import logging

# Setup logging
logging.basicConfig(
    filename='logs/manual_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load trained model
model = joblib.load('model/iris_model.pkl')

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sl', type=float, default=5.1, help='Sepal length')
parser.add_argument('--sw', type=float, default=3.5, help='Sepal width')
parser.add_argument('--pl', type=float, default=1.4, help='Petal length')
parser.add_argument('--pw', type=float, default=0.2, help='Petal width')
args = parser.parse_args()

# Create input DataFrame
input_data = np.array([[args.sl, args.sw, args.pl, args.pw]])
input_df = pd.DataFrame(input_data, columns=[
    'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# Make prediction
prediction = model.predict(input_df)[0]
print(f"Predicted class: {prediction}")

# Log result
logging.info(f"Input: {[args.sl, args.sw, args.pl, args.pw]} | Prediction: {prediction}")