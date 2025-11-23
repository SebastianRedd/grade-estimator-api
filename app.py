from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_grade

app = Flask(__name__)
CORS(app)  # allow calls from GitHub Pages  [oai_citation:3‡GitHub](https://github.com/corydolphin/flask-cors?utm_source=chatgpt.com)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    essay = data.get("essay", "")
    prompt = data.get("prompt", "")
    grade_level = data.get("grade_level", "11")

    if not essay.strip():
        return jsonify({"error": "Essay is empty"}), 400

    letter, conf, feats = predict_grade(essay, prompt, grade_level)

    # simple feedback (we'll improve later)
    feedback = []
    if feats["prompt_similarity"] < 0.2:
        feedback.append("Your response may not fully address the prompt. Re-check the main task.")
    if feats["num_words"] < 200:
        feedback.append("Your response is short. Add more evidence or explanation.")
    if feats["avg_sentence_len"] > 28:
        feedback.append("Some sentences are long. Try splitting them for clarity.")
    if not feedback:
        feedback.append("Strong writing overall. Consider adding one more concrete example.")

    return jsonify({
        "grade": letter,
        "confidence": conf,
        "features": feats,
        "feedback": feedback
    })

if __name__ == "__main__":
    app.run()