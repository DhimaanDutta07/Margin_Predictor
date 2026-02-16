from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os


MODEL_PATH = "artifacts/model/model.pkl"

FEATURES = [
    "age",
    "city_tier",
    "device",
    "membership",
    "cuisine",
    "weather",
    "day_of_week",
    "hour",
    "is_weekend",
    "distance_km",
    "prior_orders_90d",
    "avg_basket_90d",
    "rating_avg",
    "support_tickets_30d",
    "promo_type",
    "delivery_fee",
    "eta_minutes",
    "basket_value",
    "discount_amount",
    "order_total",
    "complaint_within_48h",
    "discount%",
    "amount_prior_discount",
    "delivery_fee%",
    "avg_order_total_90d",
]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="Delivery Margin Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    payload.pop("net_margin_usd", None)

    row = {f: payload.get(f, 0) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    pred = pipe.predict(X)
    return {"prediction": float(pred[0])}
