from src.data.ingest import DataLoader
from src.data.clean import DataCleaner
from src.data.split import DataSplitter
from src.features.build import FeatureBuilder
from src.model.trainer import Trainer
from src.explain.shap_report import ShapReport
from src.monitoring.evidently_report import Evidently
from src.model.tune import OptunaTuner
from src.core.config import CFG
from src.storage.s3_store import upload, download


def main():
    loader = DataLoader(CFG["raw_data"])
    df = loader.load()

    cleaner = DataCleaner()
    df = cleaner.clean(df)

    builder = FeatureBuilder()
    df = builder.build(df)

    splitter = DataSplitter()
    x_train, x_test, y_train, y_test = splitter.split(df)

    model_name = CFG.get("model_name", "xgb")

    best_params = {}
    if model_name == "xgb":
        tuner = OptunaTuner(x_train, y_train, model_name="xgb", cv=5)
        tune_out = tuner.tune(n_trials=CFG.get("optuna_trials", 40))
        best_params = tune_out["best_params"]
        print("BEST RMSE (CV):", tune_out["best_rmse"])
        print("BEST PARAMS:", best_params)

    trainer = Trainer(x_train, model_name=model_name, model_params=best_params)
    results = trainer.train(x_train, y_train, x_test, y_test)

    report = ShapReport(CFG["model_path"])
    report.get_report(x_test, out_path=CFG["shap_out"])

    evidet = Evidently(CFG["cleaned_data"], CFG["featured_data"])
    evidently_out = evidet.get_report()

    upload(CFG["model_path"], "models/model.pkl")
    upload(CFG["shap_out"], "reports/shap/summary.png")

    if isinstance(evidently_out, str) and evidently_out:
        upload(evidently_out, "reports/evidently/report.html")

    download("models/model.pkl", "artifacts/model/model_from_s3.pkl")

    print(results)


if __name__ == "__main__":
    main()
