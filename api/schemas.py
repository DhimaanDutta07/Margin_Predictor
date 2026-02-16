from pydantic import BaseModel


class PredictRequest(BaseModel):
    age: float = 22
    city_tier: float = 2
    device: str = "android"
    membership: str = "basic"
    cuisine: str = "indian"
    weather: str = "clear"
    day_of_week: str = "mon"
    hour: float = 20
    is_weekend: float = 0

    distance_km: float = 4.2
    prior_orders_90d: float = 12
    avg_basket_90d: float = 22.5
    rating_avg: float = 4.3
    support_tickets_30d: float = 0

    promo_type: str = "none"
    delivery_fee: float = 1.5
    eta_minutes: float = 32

    basket_value: float = 24.0
    discount_amount: float = 2.0
    order_total: float = 23.5

    # target if user sends it; backend will ignore it
    net_margin_usd: float = 0

    complaint_within_48h: float = 0

    # clean keys (frontend friendly)
    discount_percent: float = 0.08
    amount_prior_discount: float = 26.0
    delivery_fee_percent: float = 0.06
    avg_order_total_90d: float = 24.2


class PredictResponse(BaseModel):
    prediction: float
