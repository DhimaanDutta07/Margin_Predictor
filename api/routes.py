from fastapi import APIRouter
from .schemas import PredictRequest, PredictResponse
import pandas as pd

router = APIRouter()

# IMPORTANT: these MUST match training-time column names exactly
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
    "discount%",              # training name
    "amount_prior_discount",
    "delivery_fee%",          # training name
    "avg_order_total_90d",
]


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # imported here to avoid circular import issues
    from .main import pipe

    data = req.model_dump()

    # ignore target if user sends it
    data.pop("net_margin_usd", None)

    # Map clean keys -> training keys
    if "discount_percent" in data and "discount%" not in data:
        data["discount%"] = data.pop("discount_percent")

    if "delivery_fee_percent" in data and "delivery_fee%" not in data:
        data["delivery_fee%"] = data.pop("delivery_fee_percent")

    # build row in exact order
    row = {f: data.get(f, 0) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    pred = pipe.predict(X)
    return PredictResponse(prediction=float(pred[0]))
