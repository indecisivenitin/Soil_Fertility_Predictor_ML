# backend/predict.py
import joblib
import numpy as np

model = joblib.load("../models/best_soil_model.pkl")

def predict_fertility(input_data):
    """
    input_data: list or array of 14 values in this order:
    [NO3, NH4, P, K, SO4, B, Organic Matter, pH, Zn, Cu, Fe, Ca, Mg, Na]
    """
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return round(float(prediction[0]), 2)