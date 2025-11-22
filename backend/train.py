# backend/train.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor

# -------------------------------------------------------------
# PATHS (work on local AND Render)
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data_set.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_soil_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------
print("Applying feature engineering...")

# Total Nitrogen (NO3 + NH4)
df["N"] = df["NO3"] + df["NH4"]

# Combined NPK score
df["NPK_Sum"] = df["N"] + df["P"] + df["K"]

# Soil base saturation indicator
df["Base_Saturation"] = df["Ca"] + df["Mg"] + df["K"] - df["Na"]

# Total micronutrients effect
df["Micronutrient_Score"] = df["Zn"] + df["Cu"] + df["Fe"] + df["B"]

# -------------------------------------------------------------
# SPLIT DATA
# -------------------------------------------------------------
TARGET_COLUMN = "Vegetation Cover"
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# MODEL CANDIDATES
# -------------------------------------------------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    ),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    "LinearRegression": LinearRegression(),
    "SVR": SVR(kernel='rbf')
}

best_model = None
best_score = -999
best_name = ""

print("Training models...\n")

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        print(f"{name}: R² = {score:.4f}")

        if score > best_score:
            best_model = model
            best_score = score
            best_name = name

    except Exception as e:
        print(f"{name} FAILED → {e}")

# -------------------------------------------------------------
# SAVE BEST MODEL
# -------------------------------------------------------------
joblib.dump(best_model, MODEL_PATH)

print("\n======================================")
print(f"BEST MODEL → {best_name}  |  R² = {best_score:.4f}")
print(f"Model saved at → {MODEL_PATH}")
print("Training completed successfully!")
print("======================================")
