# backend/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Load data
df = pd.read_csv("../data/processed_data_set.csv")

# Features & target
X = df.drop("Vegetation Cover", axis=1)
y = df["Vegetation Cover"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try multiple models
models = {
"RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    "LinearRegression": LinearRegression(),
    "SVR": SVR(kernel='rbf')
}

best_model = None
best_score = 0
results = {}

print("Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test, pred)
    results[name] = score
    print(f"{name}: R² = {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# Save best model
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/best_soil_model.pkl")
print(f"\nBest model: {best_name} with R² = {best_score:.4f}")
print("Model saved as models/best_soil_model.pkl")

# Save results for README
with open("../model_results.txt", "w") as f:
    f.write(f"Best Model: {best_name}\nR² Score: {best_score:.4f}\n\nAll Results:\n")
    for n, s in results.items():
        f.write(f"{n}: {s:.4f}\n")