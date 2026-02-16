import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from .preprocessor import Preprocessor
from .models import get_model


class OptunaTuner:
    def __init__(self, x_train, y_train, model_name: str = "xgb", cv: int = 5, seed: int = 42):
        self.x_train = x_train
        self.y_train = y_train
        self.model_name = model_name.lower()
        self.cv = cv
        self.seed = seed

        self.pre = Preprocessor(x_train).build()

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        if self.model_name == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

        if self.model_name == "linreg":
            return {}

        raise ValueError("model_name must be 'xgb' or 'linreg'")

    def tune(self, n_trials: int = 30):
        def objective(trial: optuna.Trial):
            params = self._suggest_params(trial)

            pipe = Pipeline(steps=[
                ("pre", self.pre),
                ("model", get_model(self.model_name, params)),
            ])
            scores = cross_val_score(
                pipe,
                self.x_train,
                self.y_train,
                cv=self.cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            rmse = (-scores).mean()
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_rmse": study.best_value,
            "study": study,
        }
