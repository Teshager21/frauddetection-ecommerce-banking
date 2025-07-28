import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# import os

from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    """Load trained model pipeline."""
    return joblib.load(model_path)


def explain_model_with_shap(
    model_pipeline, X_sample, model_name="Model", dataset_name="ecommerce"
):
    """
    Generate SHAP summary and force plots for the given model pipeline and input data.
    """
    # Check if it's a pipeline or raw model
    if isinstance(model_pipeline, Pipeline):
        model = model_pipeline.named_steps["classifier"]
        preprocessor = model_pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()
    else:
        model = model_pipeline
        X_transformed = (
            X_sample.values if isinstance(X_sample, pd.DataFrame) else X_sample
        )
        feature_names = X_sample.columns if isinstance(X_sample, pd.DataFrame) else None

    # SHAP explanation
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    # Summary Plot
    shap.summary_plot(
        shap_values, features=X_transformed, feature_names=feature_names, show=False
    )
    plt.title(f"SHAP Summary Plot - {model_name} ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"reports/figures/shap_summary_{model_name}_{dataset_name}.png")
    plt.close()

    # Force Plot (first sample)
    shap.plots.force(shap_values[0], matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot - First Sample - {model_name} ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"reports/figures/shap_force_{model_name}_{dataset_name}.png")
    plt.close()

    return shap_values


if __name__ == "__main__":
    model = load_model("models/xgb_fraud.pkl")
    X = pd.read_csv("data/interim/interim_fraud_data.csv").sample(200, random_state=42)
    explain_model_with_shap(model, X, model_name="XGBoost", dataset_name="ecommerce")
