import pandas as pd
from src.core.config import CFG

class DataLoader():
    def __init__(self,path):
        self.path=(CFG["raw_data"])

    def load(self):
        df=pd.read_csv(self.path)
        print("dataset loaded",df.head())
        return df