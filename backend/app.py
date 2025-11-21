# backend/app.py
from flask import Flask, render_template, request
import os
import sys

# Fix import path for predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict_fertility

app = Flask(__name__)

# Smart model path – works on Render AND locally
MODEL_PATH = "/opt/render/project/src/models/best_soil_model.pkl" if os.path.exists("/opt/render") else "../models/best_soil_model.pkl"

# Auto-train only if model doesn't exist (first deploy or local)
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training XGBoost model...")
    os.system("python train.py")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test-cases")
def test_cases():
    return render_template("test-cases.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = []
        fields = ["NO3","NH4","P","K","SO4","B","OM","pH","Zn","Cu","Fe","Ca","Mg","Na"]
        for f in fields:
            val = float(request.form[f])
            data.append(val)

        fertility = predict_fertility(data)

        if fertility >= 80:
            status, color = "Excellent Fertility", "text-green-600"
        elif fertility >= 60:
            status, color = "Good Fertility", "text-emerald-600"
        elif fertility >= 40:
            status, color = "Moderate – Needs Attention", "text-yellow-600"
        else:
            status, color = "Poor – Requires Immediate Action", "text-red-600"

        return render_template("index.html", prediction=round(fertility, 2), status=status, color=color)

    except Exception as e:
        return render_template("index.html", error="Please enter valid numbers!")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)