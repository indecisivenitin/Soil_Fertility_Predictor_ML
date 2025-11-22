from flask import Flask, render_template, request, send_from_directory
import os
import joblib
from io import BytesIO
import base64

app = Flask(__name__)

# ------------------- LOAD MODEL -------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_soil_model.pkl")

if not os.path.exists(MODEL_PATH):
    print("Model not found → Training now...")
    os.system("python backend/train.py")

model = joblib.load(MODEL_PATH)
print("Model loaded → Ready for predictions!")
# --------------------------------------------------

# ------------------- RECOMMENDATION -------------------
def generate_recommendation(features, pred_pct):
    NO3, NH4, P, K, SO4, B, OM, pH, Zn, Cu, Fe, Ca, Mg, Na = features
    N = NO3 + NH4

    rec = "<ul class='space-y-3 text-lg'>"
    
    if pred_pct < 60:
        rec += "<li class='text-red-600 dark:text-red-400 font-semibold'>Critical Action Required</li>"

    # Nitrogen
    if N < 30:
        rec += "<li class='text-orange-500'>Apply <strong>Urea (46% N)</strong>: 150–200 kg/ha</li>"
    elif N < 50:
        rec += "<li class='text-yellow-500'>Apply <strong>Urea</strong>: 80–120 kg/ha</li>"
    else:
        rec += "<li class='text-green-500'>Nitrogen level is sufficient</li>"

    # Phosphorus
    if P < 40:
        rec += "<li class='text-orange-500'>Apply <strong>DAP (18-46-0)</strong>: 100–150 kg/ha</li>"
    elif P < 80:
        rec += "<li class='text-yellow-500'>Apply <strong>DAP</strong>: 50–80 kg/ha</li>"
    else:
        rec += "<li class='text-green-500'>Phosphorus is adequate</li>"

    # Potassium
    if K < 100:
        rec += "<li class='text-orange-500'>Apply <strong>Muriate of Potash (60% K)</strong>: 80–120 kg/ha</li>"
    else:
        rec += "<li class='text-green-500'>Potassium is sufficient</li>"

    # Organic Matter
    if OM < 2:
        rec += "<li class='text-orange-500'>Add <strong>FYM</strong>: 10–15 tons/ha</li>"
    elif OM < 4:
        rec += "<li class='text-yellow-500'>Add <strong>FYM/Compost</strong>: 5–8 tons/ha</li>"

    # pH
    if pH > 7.8:
        rec += "<li class='text-orange-500'>Apply <strong>Gypsum/Sulfur</strong> to lower pH</li>"
    elif pH < 6.0:
        rec += "<li class='text-orange-500'>Apply <strong>Lime</strong>: 2–5 tons/ha</li>"

    # Micronutrients
    if Zn < 1.5:
        rec += "<li class='text-orange-500'>Apply <strong>Zinc Sulphate</strong>: 25 kg/ha</li>"
    if Fe < 5:
        rec += "<li class='text-orange-500'>Apply <strong>Ferrous Sulphate</strong>: 20–25 kg/ha</li>"

    rec += "<li class='text-emerald-600 dark:text-emerald-400 mt-4 font-bold'>Target: 85%+ fertility in one season</li>"
    rec += "</ul>"
    return rec

# ------------------- PDF GENERATOR -------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(features, pred_pct, status, recommendation):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.8*inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("SOIL FERTILITY REPORT", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<font size=14>Predicted Fertility: <b>{pred_pct}%</b> → {status}</font>", styles['Normal']))
    story.append(Spacer(1, 30))

    data = [
        ["Parameter", "Value", "Unit"],
        ["NO₃-N", f"{features[0]:.2f}", "ppm"],
        ["NH₄-N", f"{features[1]:.2f}", "ppm"],
        ["Phosphorus", f"{features[2]:.2f}", "ppm"],
        ["Potassium", f"{features[3]:.2f}", "ppm"],
        ["Organic Matter", f"{features[6]:.2f}", "%"],
        ["pH", f"{features[7]:.1f}", ""],
        ["Zinc", f"{features[8]:.2f}", "ppm"]
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#22c55e")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f0fdf4"))
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Fertilizer Recommendations:</b>", styles['Heading2']))
    story.append(Paragraph(recommendation, styles['Normal']))

    doc.build(story)
    pdf_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return f"data:application/pdf;base64,{pdf_base64}"

# ------------------- ROUTES -------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test-cases")
def test_cases():
    return render_template("test-cases.html")

@app.route('/manifest.json')
def manifest():
    return send_from_directory('../static', 'manifest.json')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['NO3']),
            float(request.form['NH4']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['SO4']),
            float(request.form['B']),
            float(request.form['OM']),
            float(request.form['pH']),
            float(request.form['Zn']),
            float(request.form['Cu']),
            float(request.form['Fe']),
            float(request.form['Ca']),
            float(request.form['Mg']),
            float(request.form['Na'])
        ]

        prediction_pct = round(float(model.predict([features])[0]), 2)

        if prediction_pct >= 90:
            status, color = "Ultra Fertile", "text-emerald-600 dark:text-emerald-400"
        elif prediction_pct >= 75:
            status, color = "Very Good", "text-green-600 dark:text-green-400"
        elif prediction_pct >= 60:
            status, color = "Good", "text-yellow-600 dark:text-yellow-400"
        elif prediction_pct >= 40:
            status, color = "Needs Improvement", "text-orange-600 dark:text-orange-400"
        else:
            status, color = "Poor / Deficient", "text-red-600 dark:text-red-400"

        recommendation = generate_recommendation(features, prediction_pct)
        pdf_report = generate_pdf(features, prediction_pct, status, recommendation)

        return render_template('index.html',
                               prediction=prediction_pct,
                               status=status,
                               color=color,
                               recommendation=recommendation,
                               pdf_report=pdf_report,
                               features=features)

    except Exception as e:
        print("ERROR:", str(e))
        return render_template('index.html', error="Invalid input! Please fill all fields correctly.")

if __name__ == "__main__":
    app.run(debug=True)