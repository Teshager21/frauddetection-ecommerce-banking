# src/utils/explainability.py
import shap
import matplotlib.pyplot as plt
import pandas as pd

# import numpy as np
import joblib
import os


def explain_model_with_shap(model, X, model_name="model", dataset_name="dataset"):
    """
    Generates SHAP summary and force plots for a given model and feature set.

    Args:
        model: A trained model (e.g., XGBClassifier).
        X (pd.DataFrame): The feature data.
        model_name (str): Name of the model (for saving figures).
        dataset_name (str): Name of dataset (for saving figures).
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Plot summary
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    summary_path = f"reports/figures/shap_summary_{model_name}_{dataset_name}.png"
    plt.savefig(summary_path)
    print(f"SHAP summary plot saved to: {summary_path}")
    plt.close()

    # Optional: Show force plot for a single sample
    sample_idx = 0
    force_plot = shap.plots.force(shap_values[sample_idx], matplotlib=False)
    force_html_path = f"reports/figures/shap_force_{model_name}_{dataset_name}.html"
    with open(force_html_path, "w") as f:
        f.write(force_plot.html())
    print(f"SHAP force plot saved to: {force_html_path} (open in browser)")


# Example usage (run this script directly)
if __name__ == "__main__":
    # Load model and features
    model = joblib.load("models/xgb_model.joblib")
    X = pd.read_parquet("data/processed/X_test.parquet")

    explain_model_with_shap(model, X, model_name="XGBoost", dataset_name="ecommerce")
