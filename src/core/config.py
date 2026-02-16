CFG = {
    "raw_data": "data/raw/delivery_dual_task_clean.csv",
    "cleaned_data": "data/processed/cleaned.csv",
    "featured_data": "data/processed/featured.csv",

    "model_path": "artifacts/model/model.pkl",
    "shap_out": "artifacts/shap/summary.png",

    "target": "net_margin_usd",
    "test_size": 0.2,
    "random_state": 42,

    "model_name": "xgb",
    "optuna_trials": 40,
}
