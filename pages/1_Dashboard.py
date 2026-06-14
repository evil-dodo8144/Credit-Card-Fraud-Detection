import streamlit as st
import pandas as pd
import json

from utils.plots import (
    fraud_distribution_chart,
    amount_distribution_chart
)

st.title("📊 Dashboard")


@st.cache_data
def load_data():

    return pd.read_csv(
        "dataset/creditcard.csv"
    )


df = load_data()

# -----------------------------------
# Load Metrics
# -----------------------------------

with open(
    "reports/metrics.json",
    "r"
) as file:

    metrics = json.load(file)

rf_metrics = metrics[
    "Random Forest"
]

# -----------------------------------
# Statistics
# -----------------------------------

total_transactions = len(df)

fraud_count = (
    df["Class"] == 1
).sum()

genuine_count = (
    df["Class"] == 0
).sum()

fraud_percentage = (
    fraud_count /
    total_transactions
) * 100

# -----------------------------------
# Metrics Row
# -----------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Cases",
    fraud_count
)

col3.metric(
    "Fraud %",
    f"{fraud_percentage:.3f}%"
)

col4.metric(
    "ROC-AUC",
    rf_metrics["roc_auc"]
)

# -----------------------------------
# Charts
# -----------------------------------

st.markdown("---")

left, right = st.columns(2)

with left:

    st.plotly_chart(
        fraud_distribution_chart(
            df
        ),
        use_container_width=True
    )

with right:

    st.plotly_chart(
        amount_distribution_chart(
            df
        ),
        use_container_width=True
    )

# -----------------------------------
# Dataset Preview
# -----------------------------------

st.markdown("---")

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)

# -----------------------------------
# Dataset Shape
# -----------------------------------

st.subheader(
    "Dataset Information"
)

col1, col2 = st.columns(2)

col1.metric(
    "Rows",
    df.shape[0]
)

col2.metric(
    "Columns",
    df.shape[1]
)

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection | AIML Project"
)