import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide"
)


# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():

    decision_tree = joblib.load(
        MODEL_DIR / "decision_tree_model.pkl"
    )

    random_forest = joblib.load(
        MODEL_DIR / "random_forest_model.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "feature_columns.pkl"
    )

    return decision_tree, random_forest, feature_columns


# -----------------------------
# SHAP Explainability Function
# -----------------------------
def explain_transaction(model, transaction):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(transaction)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif len(shap_values.shape) == 3:

        values = shap_values[0, :, 1]

    else:

        values = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": transaction.columns,
        "SHAP Value": values
    })

    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Value"].abs()
    )

    explanation_df = explanation_df.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    return explanation_df


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    return pd.read_csv(
        BASE_DIR /"sample_data"/
        "fraudlens_features.csv"
    )


# -----------------------------
# Main Application
# -----------------------------
def main():

    st.title("🔍 FraudLens")

    st.subheader(
        "Explainable AI Fraud Detection & Investigation System"
    )
    st.info(
    "FraudLens analyzes transaction data using Decision Tree "
    "and Random Forest models, estimates fraud risk, and uses "
    "SHAP to explain which features influenced the prediction."
)

    st.write(
        "FraudLens uses machine learning to identify "
        "suspicious transactions and assign fraud-risk scores."
    )

    st.divider()

    # -----------------------------
    # Load Data and Models
    # -----------------------------
    try:

        decision_tree, random_forest, feature_columns = load_models()

        df = load_data()

    except Exception as e:

        st.error("Error loading project files.")

        st.exception(e)

        st.stop()

    # -----------------------------
    # Dataset Overview
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Transactions",
            len(df)
        )

    with col2:

        st.metric(
            "Features Used",
            len(feature_columns)
        )

    with col3:

        st.metric(
            "Model",
            "Random Forest"
        )

    st.divider()

    # -----------------------------
    # Transaction Investigation
    # -----------------------------
    st.header("Transaction Investigation")

    transaction_index = st.number_input(
        "Enter Transaction Row Number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    # Select transaction
    transaction = df[feature_columns].iloc[
        [transaction_index]
    ].copy()

    # -----------------------------
    # Predictions
    # -----------------------------
    dt_prediction = decision_tree.predict(
        transaction
    )[0]

    rf_prediction = random_forest.predict(
        transaction
    )[0]

    fraud_probability = random_forest.predict_proba(
        transaction
    )[0][1]

    # -----------------------------
    # Risk Level
    # -----------------------------
    if fraud_probability >= 0.75:

        risk_level = "HIGH"

    elif fraud_probability >= 0.40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # -----------------------------
    # Prediction Results
    # -----------------------------
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Decision Tree",
            "Fraud" if dt_prediction == 1
            else "Legitimate"
        )

    with col2:

        st.metric(
            "Random Forest",
            "Fraud" if rf_prediction == 1
            else "Legitimate"
        )

    with col3:

        st.metric(
            "Fraud Score",
            f"{fraud_probability * 100:.2f}%"
        )

    # -----------------------------
    # Risk Display
    # -----------------------------
    st.subheader("Risk Level")

    if risk_level == "HIGH":

        st.error(
            "🚨 HIGH RISK — Transaction requires investigation."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM RISK — Transaction should be reviewed."
        )

    else:

        st.success(
            "✅ LOW RISK — Transaction appears legitimate."
        )

    # -----------------------------
    # Transaction Details
    # -----------------------------
    st.subheader("Transaction Details")

    st.dataframe(
        transaction,
        use_container_width=True
    )

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.divider()

    st.header(
        "🔍 Explainable AI — Transaction Investigation"
    )

    st.write(
        "This section shows which features had the strongest "
        "influence on the Random Forest prediction."
    )

    try:

        explanation_df = explain_transaction(
            random_forest,
            transaction
        )

        st.subheader("Feature Contributions")

        st.dataframe(
            explanation_df,
            use_container_width=True
        )

        # Select top 10 features
        top_features = explanation_df.head(10).copy()

        # Sort for horizontal bar chart
        top_features = top_features.sort_values(
            by="SHAP Value"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            top_features["Feature"],
            top_features["SHAP Value"]
        )

        ax.set_xlabel("SHAP Value")

        ax.set_title(
            "Top Features Influencing Fraud Prediction"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    except Exception as e:

        st.error(
            "Unable to generate SHAP explanation."
        )

        st.exception(e)
        # -----------------------------
    # Download Investigation Report
    # -----------------------------
    st.divider()
    st.header("Download Investigation Report")

    report = pd.DataFrame({
        "Transaction Row": [int(transaction_index)],
        "Decision Tree Prediction": [
            "Fraud" if dt_prediction == 1 else "Legitimate"
        ],
        "Random Forest Prediction": [
            "Fraud" if rf_prediction == 1 else "Legitimate"
        ],
        "Fraud Score (%)": [
            round(fraud_probability * 100, 2)
        ],
        "Risk Level": [risk_level]
    })

    st.download_button(
        label="Download Investigation Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name=f"fraudlens_report_row_{int(transaction_index)}.csv",
        mime="text/csv"
    )

    # -----------------------------
    # Footer
    # -----------------------------
    st.divider()

    st.caption(
        "FraudLens | Machine Learning Fraud Detection Project"
    )


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":

    main()