import pandas as pd
import os

class FeatureBuilder():
    def __init__(self):
        pass

    def build(self,df):
        df=df.copy()
        df["discount%"]=df["discount_amount"]/df["order_total"]*100
        df["amount_prior_discount"]=df["order_total"]+df["discount_amount"]
        df["delivery_fee%"]=df["delivery_fee"]/df["order_total"]*100
        deliv_fee=(df["delivery_fee%"]/100)*df["avg_basket_90d"]
        diss=(df["discount%"]/100)*df["avg_basket_90d"]
        df["avg_order_total_90d"]=df["avg_basket_90d"]+deliv_fee-diss

        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/featured.csv", index=False)
        print("Features built and saved → data/processed/featured.csv")
        return df