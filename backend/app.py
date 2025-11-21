# backend/app.py
from flask import Flask, render_template, request
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import predict_fertility

app = Flask(__name__)

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
        for field in ["NO3", "NH4", "P", "K", "SO4", "B", "OM", "pH", "Zn", "Cu", "Fe", "Ca", "Mg", "Na"]:
            val = float(request.form[field])
            data.append(val)
        
        fertility = predict_fertility(data)
        
        # Simple recommendation
        if fertility >= 80:
            status = "Excellent Fertility"
            color = "text-green-600"
        elif fertility >= 60:
            status = "Good Fertility"
            color = "text-emerald-600"
        elif fertility >= 40:
            status = "Moderate – Needs Attention"
            color = "text-yellow-600"
        else:
            status = "Poor – Requires Immediate Action"
            color = "text-red-600"
            
        return render_template("index.html", prediction=fertility, status=status, color=color)
    except:
        return render_template("index.html", error="Please enter valid numbers!")

if __name__ == "__main__":
    # First train the model if not exists
        print("Training model for first deploy...")
    os.system("python train.py")
        print("Training model first...")
        os.system("python backend/train.py")
    app.run(debug=True)