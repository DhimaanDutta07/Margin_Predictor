from sklearn.pipeline import Pipeline
import joblib
import mlflow
import os

from .models import get_model
from .preprocessor import Preprocessor
from .evaluate import Metrics


class Trainer:
    def __init__(self, x_train, model_name: str = "xgb", model_params: dict | None = None):
        self.model_name = model_name.lower()
        self.model_params = model_params or {}

        self.pipe = Pipeline(steps=[
            ("pre", Preprocessor(x_train).build()),
            ("model", get_model(self.model_name, self.model_params)),
        ])

    def train(self, x_train, y_train, x_test, y_test):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("regression")

        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_param("model_name", self.model_name)
            if self.model_params:
                mlflow.log_params(self.model_params)

            self.pipe.fit(x_train, y_train)

            os.makedirs("artifacts/model", exist_ok=True)
            model_path = "artifacts/model/model.pkl"
            joblib.dump(self.pipe, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

            y_pred = self.pipe.predict(x_test)
            metrics = Metrics().get_metrics(y_test, y_pred)

            mlflow.log_metric("mse", float(metrics["mse"]))
            mlflow.log_metric("mae", float(metrics["mae"]))
            mlflow.log_metric("r2", float(metrics["r2"]))

            return metrics
