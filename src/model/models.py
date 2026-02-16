from sklearn.linear_model import LinearRegression

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def get_model(name: str, params: dict | None = None):
    name = name.lower()
    params = params or {}

    if name == "linreg":
        return LinearRegression(**params)

    if name == "xgb":
        if XGBRegressor is None:
            raise ImportError("pip install xgboost")

        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

        defaults.update(params)
        return XGBRegressor(**defaults)

    raise ValueError("Invalid model name. Use: 'linreg' or 'xgb'")
