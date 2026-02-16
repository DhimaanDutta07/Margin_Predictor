import os
import joblib
import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataSummaryPreset, DataDriftPreset, ClassificationPreset

class Evidently():
    def __init__(self,cur_path,ref_path):
        self.curr_path="data/processed/cleaned.csv"
        self.ref_path="data/processed/featured.csv"

    def get_report(self):
        ref_ds = pd.read_csv(self.ref_path)
        cur_ds = pd.read_csv(self.curr_path)

        rep = Report([DataSummaryPreset(), DataDriftPreset()])
        snap = rep.run(cur_ds, ref_ds)
        out_dir="artifacts/monitoring"
        out_path = os.path.join(out_dir, "evidently_report.html")
        snap.save_html(out_path)
        return out_path