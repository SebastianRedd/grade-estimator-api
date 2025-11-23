from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from features import basic_features
from feedback import generate_feedback

app = Flask(__name__)
CORS(app)

model = load("grade_model.joblib")

@app.route("/estimate", methods=["POST"])
def estimate():
    data = request.get_json()

    essay = data.get("essay","")
    prompt = data.get("prompt","")
    grade_level = data.get("grade_level","11")

    feats = basic_features(essay, prompt)
    feats["grade_level"] = grade_level

    pred = model.predict([feats])[0]
    probs = model.predict_proba([feats])[0]
    conf = float(max(probs))

    tips = generate_feedback(feats, grade_level)

    return jsonify({
        "grade": pred,
        "confidence": round(conf, 3),
        "features": feats,
        "feedback": tips
    })

@app.route("/")
def home():
    return "Grade Estimator API is running."

if __name__ == "__main__":
    app.run()