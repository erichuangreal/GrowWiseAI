# Use pandas for high-level data manipulation and loading the CSV
import pandas as pd

# Use numpy for low-level numerical operations and array handling
import numpy as np

# A function in sklearn library that can be used to split dataset into training, validation, and test sets
from sklearn.model_selection import train_test_split

# The Random Forest Base that we need to build our model
from sklearn.ensemble import RandomForestClassifier

# A function in sklearn to determine the hit/miss rate of our model's predictions 
# based on the actual labels associated with each subset of features
from sklearn.metrics import accuracy_score

# Load the dataset from the file
df = pd.read_csv('backend/Features&Labels.csv')

# Define features and target

#Pulls only the values from these headers
features = ['elevation', 'temperature', 'humidity', 'soil_TN', 'soil_TP', 'soil_AP', 'soil_AN']
X = df[features]

# Our labels
y = df['health_class']

# Split into train (60%), val (20%), test (20%)
# Separate the training dataset from the main dataset first
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Split the remaining dataset in half for validation set and test set
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(
    # Use 100 individual decision trees; more trees increase accuracy but take more time/memory
    n_estimators=100,
    
    # Limit trees to 5 levels deep to keep the model simple and prevent it from over-learning noise
    max_depth=5,
    
    # A node must have at least 20 samples to split; helps avoid creating tiny, unreliable branches
    min_samples_split=20,
    
    # Each 'leaf' (end of a branch) must have 10 samples; ensures predictions are based on a group, not an outlier
    min_samples_leaf=10,
    
    # Set a 'seed' so the random parts of the model stay the same every time we run the code
    # Helps us evaluate the model and debug by providing consistent testing
    random_state=42,
    
    # Use all available CPU cores to speed up the training process (-1 means 'use everything')
    n_jobs=-1
)

# Our model trains on the training dataset
rf_model.fit(X_train, y_train)

# Make predictions on all sets
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Calculate accuracies
train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

# Evaluate the model
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

# Feature importance
for feature, importance in zip(features, rf_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Save the model, keeps the parameters available and allows our frontend to use the model to make predictions
import joblib
joblib.dump(rf_model, 'backend/tree_health_rf_model.pkl')