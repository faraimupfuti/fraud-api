"""
Zimswitch Fraud Detection API
=================================================================
A small FastAPI service that wraps the trained Random Forest fraud
model. This is the backend: it does the actual model inference. The
Streamlit app is a separate frontend that calls this API over HTTP —
the same client/server split a real payments fraud system would use,
so the model can be scaled, monitored, and updated independently of
whatever is presenting it to a user.

Run locally:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs
(FastAPI generates this automatically).

Deploy on Render (free tier):
    1. Push this `api/` folder to a GitHub repo.
    2. On render.com: New -> Web Service -> connect the repo.
    3. Build command:  pip install -r requirements.txt
       Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
    4. Deploy. You'll get a URL like https://your-app.onrender.com
"""

import io
import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import (
    build_single_row, build_checklist, haversine_km, CITIES,
    engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Zimswitch Fraud Detection API",
    description=(
        "Unofficial portfolio project — not affiliated with, endorsed by, or built for "
        "Zimswitch. Scores synthetic payment-switch transactions for fraud in real time."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different domain) to call this API.
# Fine for a portfolio demo; a production service would restrict this to known origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rf_pipe = None
metadata = None


@app.on_event("startup")
def load_model():
    global rf_pipe, metadata
    rf_pipe = joblib.load(MODELS_DIR / "rf_pipeline.joblib")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, example=50.0, description="Transaction amount")
    currency: str = Field("USD", example="USD", description="USD or ZWG")
    channel: str = Field(..., example="POS",
                          description="POS, ATM, ZIPIT, Mobile Money, or Internet Banking")
    mcc: str = Field(..., example="5411", description="Merchant category code")
    city_now: str = Field(..., example="Harare", description="City where this transaction is happening")
    city_prev: str = Field(..., example="Harare", description="City of this card's previous transaction")
    hour: int = Field(..., ge=0, le=23, example=14, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="0 = Monday, 6 = Sunday")
    minutes_since_prev: float = Field(..., gt=0, example=120,
                                       description="Minutes since this card's previous transaction")
    txn_count_1h: int = Field(0, ge=0, example=0, description="Transactions on this card in the last hour")
    txn_count_24h: int = Field(1, ge=0, example=1, description="Transactions on this card in the last 24 hours")
    typical_amount: float = Field(..., gt=0, example=45.0,
                                   description="What this card usually spends per transaction")


class CheckItem(BaseModel):
    label: str
    value: str
    flag: bool
    detail: str


class TransactionResponse(BaseModel):
    is_fraud: bool
    probability: float
    verdict: str
    km_apart: float
    implied_speed_kmh: float
    checks: List[CheckItem]


@app.get("/")
def root():
    return {
        "service": "Zimswitch Fraud Detection API",
        "status": "ok" if rf_pipe is not None else "model not loaded",
        "docs": "/docs",
        "disclaimer": "Unofficial portfolio project. Not affiliated with Zimswitch.",
    }


@app.get("/health")
def health():
    return {"status": "ok" if rf_pipe is not None else "loading"}


@app.get("/metadata")
def get_metadata():
    """Model evaluation metrics, feature importances, and sample flagged transactions."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded yet")
    return metadata


@app.get("/cities")
def get_cities():
    """Known cities and coordinates, for building a city picker client-side."""
    return {"cities": list(CITIES.keys())}


@app.post("/predict", response_model=TransactionResponse)
def predict(txn: TransactionRequest):
    if rf_pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, try again shortly")

    if txn.city_now not in CITIES or txn.city_prev not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown city. Known cities: {', '.join(CITIES.keys())}",
        )

    lat1, lon1 = CITIES[txn.city_now]
    lat2, lon2 = CITIES[txn.city_prev]
    km_apart = float(haversine_km(lat1, lon1, lat2, lon2))

    row = build_single_row(
        amount=txn.amount, hour=txn.hour, day_of_week=txn.day_of_week,
        minutes_since_prev=txn.minutes_since_prev, km_from_prev=km_apart,
        txn_count_1h=txn.txn_count_1h, txn_count_24h=txn.txn_count_24h,
        channel=txn.channel, mcc=txn.mcc, currency=txn.currency,
        avg_amount=txn.typical_amount, std_amount=txn.typical_amount * 0.4,
    )

    probability = float(rf_pipe.predict_proba(row)[0, 1])
    is_fraud = probability >= 0.5

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    checks = build_checklist(
        amount=txn.amount, hour=txn.hour, day_name=day_names[txn.day_of_week],
        minutes_since_prev=txn.minutes_since_prev, km_apart=km_apart,
        txn_count_1h=txn.txn_count_1h, txn_count_24h=txn.txn_count_24h,
        typical_amount=txn.typical_amount,
    )

    hours_gap = max(txn.minutes_since_prev / 60, 1 / 60)
    implied_speed = min(km_apart / hours_gap, 2000)

    return TransactionResponse(
        is_fraud=is_fraud,
        probability=probability,
        verdict="FRAUDULENT" if is_fraud else "CLEAN",
        km_apart=km_apart,
        implied_speed_kmh=implied_speed,
        checks=checks,
    )


REQUIRED_BATCH_COLUMNS = {
    "card_id", "timestamp", "amount", "currency", "channel", "mcc", "latitude", "longitude",
}


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV of raw transactions and get a fraud prediction for every
    row. Unlike /predict, this computes each card's behavioral features
    (velocity, travel speed, spend deviation) directly from the
    transaction history in the file, the same way the model was trained —
    so a card's 5th row in the file benefits from seeing its first 4.

    Required columns: card_id, timestamp, amount, currency, channel, mcc,
    latitude, longitude. An optional transaction_id column is carried
    through to the output for reference.
    """
    if rf_pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet, try again shortly")

    contents = await file.read()
    try:
        df = pd.read_csv(
            io.BytesIO(contents),
            dtype={"mcc": str, "card_id": str, "currency": str, "channel": str},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read this as a CSV file: {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file has no rows")

    missing = REQUIRED_BATCH_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(f"Missing required column(s): {sorted(missing)}. "
                    f"Required columns: {sorted(REQUIRED_BATCH_COLUMNS)}"),
        )

    try:
        df_feat = engineer_features(df)
        X = df_feat[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        probabilities = rf_pipe.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not score this file: {e}")

    df_feat["fraud_probability"] = probabilities
    df_feat["verdict"] = np.where(probabilities >= 0.5, "FRAUDULENT", "CLEAN")
    has_txn_id = "transaction_id" in df_feat.columns

    results = []
    for i, row in df_feat.iterrows():
        results.append({
            "transaction_id": str(row["transaction_id"]) if has_txn_id else str(i),
            "card_id": str(row["card_id"]),
            "timestamp": str(row["timestamp"]),
            "amount": float(row["amount"]),
            "currency": str(row["currency"]),
            "channel": str(row["channel"]),
            "fraud_probability": float(row["fraud_probability"]),
            "verdict": str(row["verdict"]),
        })

    n_flagged = int((probabilities >= 0.5).sum())
    return {
        "n_transactions": int(len(df_feat)),
        "n_flagged": n_flagged,
        "flagged_rate": n_flagged / len(df_feat),
        "results": results,
    }
