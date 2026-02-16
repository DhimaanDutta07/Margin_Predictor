import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.config import CFG


class DataSplitter:
    def __init__(self, target_col: str = "net_margin_usd", test_size: float = 0.20, random_state: int = 42):
        self.target_col = CFG["target"]
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame):
        df = df.copy()

        X = df.drop(self.target_col, axis=1)
        y = pd.to_numeric(df[self.target_col], errors="coerce")

        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print("created: x_train,x_test,y_train,y_test")
        return x_train, x_test, y_train, y_test
