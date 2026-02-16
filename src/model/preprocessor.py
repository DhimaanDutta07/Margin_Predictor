import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self, X: pd.DataFrame):
        self.X = X.copy()
        self.num_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in self.X.columns if c not in self.num_cols]
        self.pre = None

    def build(self) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        self.pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols),
                ("cat", cat_pipe, self.cat_cols),
            ],
            remainder="drop",
        )
        return self.pre

    def fit(self) -> "Preprocessor":
        if self.pre is None:
            self.build()
        self.pre.fit(self.X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pre is None:
            raise RuntimeError("Preprocessor not built/fitted. Call fit() first.")
        Xt = self.pre.transform(X)
        cols = self.num_cols + self.cat_cols
        return pd.DataFrame(Xt, columns=cols)

    def fit_transform(self) -> pd.DataFrame:
        if self.pre is None:
            self.build()
        Xt = self.pre.fit_transform(self.X)
        cols = self.num_cols + self.cat_cols
        return pd.DataFrame(Xt, columns=cols)

    def save(self, df: pd.DataFrame, out_path: str = "data/processed/encoded_x_scaled.csv") -> str:
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    def fit_transform_and_save(self, out_path: str = "data/processed/encoded_x_scaled.csv") -> pd.DataFrame:
        df_out = self.fit_transform()
        self.save(df_out, out_path)
        return df_out
