FraudLens — Explainable AI Fraud Detection \& Investigation System

Overview

FraudLens is a machine learning project designed to identify potentially fraudulent transactions, estimate fraud risk, and explain model predictions using SHAP (SHapley Additive exPlanations).

Features

* Fraud prediction using Decision Tree and Random Forest models
* Fraud probability scoring
* Low, medium, and high risk classification
* SHAP-based feature contribution analysis
* Interactive Streamlit dashboard
* Transaction investigation by row number
* CSV export of investigation results

Technology Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* SHAP
* Joblib

Project Structure

Fraud detection/
├── data/
├── dashboard/
│   └── app.py
├── models/
├── notebooks/
├── src/
├── matlab/
├── README.md
└── requirements.txt

Installation

1. Install Python.
2. Download or clone the project repository.
3. Open a terminal in the project directory.
4. Install dependencies:

python -m pip install -r requirements.txt

Run the Dashboard

streamlit run dashboard/app.py

How to Use

1. Open the FraudLens dashboard.
2. Enter a transaction row number.
3. Review the Decision Tree and Random Forest predictions.
4. Check the fraud score and assigned risk level.
5. Review the SHAP feature contributions and bar chart.
6. Export the investigation report as a CSV file.

Model Explainability

FraudLens uses SHAP to show how individual input features contribute to a model prediction. These contributions help users investigate a transaction rather than relying only on a prediction score.

Disclaimer

FraudLens is an educational portfolio project. Predictions are not proof of fraud and should not be used as the sole basis for real financial decisions.

Author

BSc Data Science Student

