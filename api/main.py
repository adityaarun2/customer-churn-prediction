# api/main.py
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Radar API", version="1.0")

# Load the same pipeline as the UI
PIPE = joblib.load("models/pipeline.joblib")

class PredictIn(BaseModel):
    features: dict  # one record

class PredictBatchIn(BaseModel):
    records: list[dict]  # many records

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictIn):
    df = pd.DataFrame([req.features])
    proba = float(PIPE.predict_proba(df)[:, 1][0])
    return {"churn_probability": proba}

@app.post("/predict-batch")
def predict_batch(req: PredictBatchIn):
    df = pd.DataFrame(req.records)
    proba = PIPE.predict_proba(df)[:, 1].tolist()
    return {"churn_probability": proba}
