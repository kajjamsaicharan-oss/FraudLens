import shap
import pandas as pd


def explain_prediction(model, X_background, X_row):

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_row)

    # Handle binary classification output
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    # Create explanation table
    explanation_df = pd.DataFrame({
        "Feature": X_row.columns,
        "SHAP Value": values
    })

    # Calculate absolute contribution
    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Value"].abs()
    )

    # Sort by importance
    explanation_df = explanation_df.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    return explanation_df