import os
import joblib
import shap
import matplotlib.pyplot as plt


class ShapReport:
    def __init__(self, model_path: str = "artifacts/model/model.pkl"):
        self.model_path = model_path
        self.pipe = joblib.load(model_path)
        self.pre = self.pipe.named_steps["pre"]
        self.model = self.pipe.named_steps["model"]

    def get_report(self, x_test, out_path: str = "artifacts/shap/summary.png"):
        os.makedirs("artifacts/shap", exist_ok=True)

        x_test_t = self.pre.transform(x_test)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(x_test_t)

        plt.figure()
        shap.summary_plot(shap_values, x_test_t, show=False)
        plt.savefig(out_path, bbox_inches="tight", dpi=250)
        plt.close()

        return out_path
