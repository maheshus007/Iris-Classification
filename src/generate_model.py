import joblib  # For saving the model as .pkl
import pandas as pd  # For loading the dataset
from sklearn.model_selection import train_test_split  # To split the data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # The classifier
from sklearn.metrics import accuracy_score  # To check the accuracy of the model

# Step 1: Load the Iris dataset from the CSV file
data = pd.read_csv('../data/iris.csv')  # Ensure that the CSV file is present in the data folder

# Display the first few rows of the dataset
print("Dataset Loaded:")
print(data.head())

# Step 2: Prepare features (X) and target (y)
X = data.drop(['Id', 'Species'], axis=1)  # Drop 'Id' and 'Species' columns
y = data['Species']  # The target variable (species)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Step 6: Save the trained model as a .pkl file
joblib.dump(model, 'model/iris_model.pkl')

print("Model saved as 'model/iris_model.pkl'")
