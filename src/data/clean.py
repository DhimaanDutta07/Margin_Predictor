import os
import pandas as pd


class DataCleaner:
    def __init__(self):
        pass

    def clean(self, df):
        df = df.copy()
        df=df.drop("order_id",axis=1)
        print("NaN values per column:\n", df.isna().sum())
        print("Duplicate rows:", df.duplicated().sum())

        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/cleaned.csv", index=False)

        return df
