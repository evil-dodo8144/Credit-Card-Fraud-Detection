import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from utils.plots import (
    confusion_matrix_plot,
    roc_curve_plot
)

st.title("⚖️ Model Comparison")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_data
def load_data():

    url = "https://drive.google.com/file/d/18vFxh2HBkXdEG5D9I4PmQ2YgPYvHNG75/view?usp=sharing"

    return pd.read_csv(url)

df =  load_data()

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------
# Load Models
# ----------------------------------

rf = joblib.load(
    "models/random_forest.pkl"
)

lr = joblib.load(
    "models/logistic_regression.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

X_test = scaler.transform(X_test)

# ----------------------------------
# Threshold
# ----------------------------------

threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50
)

# ----------------------------------
# Model Selection
# ----------------------------------

selected_model = st.selectbox(
    "Choose Model",
    [
        "Random Forest",
        "Logistic Regression"
    ]
)

model = (
    rf if selected_model ==
    "Random Forest"
    else lr
)

# ----------------------------------
# Predictions
# ----------------------------------

y_prob = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_prob >= threshold
).astype(int)

# ----------------------------------
# Metrics
# ----------------------------------

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

# ----------------------------------
# Display Metrics
# ----------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Precision",
    f"{precision:.4f}"
)

col2.metric(
    "Recall",
    f"{recall:.4f}"
)

col3.metric(
    "F1 Score",
    f"{f1:.4f}"
)

col4.metric(
    "ROC-AUC",
    f"{roc_auc:.4f}"
)

# ----------------------------------
# Confusion Matrix
# ----------------------------------

st.markdown("---")

st.subheader(
    "Confusion Matrix"
)

st.plotly_chart(
    confusion_matrix_plot(
        y_test,
        y_pred
    ),
    use_container_width=True
)

# ----------------------------------
# ROC Curve
# ----------------------------------

st.markdown("---")

st.subheader(
    "ROC Curve"
)

st.plotly_chart(
    roc_curve_plot(
        y_test,
        y_prob
    ),
    use_container_width=True
)

# ----------------------------------
# Threshold Insights
# ----------------------------------

st.markdown("---")

st.subheader(
    "Threshold Analysis"
)

st.info(
    """
Lower Threshold → Higher Recall, More False Positives

Higher Threshold → Higher Precision, More Fraud Misses
"""
)