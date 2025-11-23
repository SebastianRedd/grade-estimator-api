from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from features import basic_features

app = FastAPI()

# allow your GitHub Pages site to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can lock to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    assignment: str
    prompt: str = ""
    grade_level: str = "11"
    assignment_type: str = "essay"

# load model if exists
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
        # fallback if model not trained yet
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