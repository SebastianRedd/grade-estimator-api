from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from features import basic_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    assignment: str
    prompt: str = ""
    grade_level: str = "11"
    assignment_type: str = "essay"

try:
    model = load("grade_model.joblib")
except:
    model = None

@app.get("/")
def root():
    return {"status":"ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(req: PredictRequest):
    feats = basic_features(req.assignment, req.prompt)
    feats["grade_level"] = req.grade_level
    feats["assignment_type"] = req.assignment_type

    if model is None:
        return {
            "letter": "B+",
            "confidence": 0.50,
            "features": feats,
            "note": "Model not trained yet; returning fallback."
        }

    pred = model.predict([feats])[0]
    probs = model.predict_proba([feats])[0]
    conf = float(max(probs))

    return {
        "letter": pred,
        "confidence": round(conf, 3),
        "features": feats
    }