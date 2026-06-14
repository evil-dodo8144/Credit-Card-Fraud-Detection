import streamlit as st
import pandas as pd
import joblib

from utils.plots import (
    probability_distribution
)

st.title("🔍 Batch Fraud Prediction")

st.write(
    "Upload a CSV file containing transaction data."
)

# ----------------------------------
# Upload CSV
# ----------------------------------

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            "File Uploaded Successfully"
        )

        st.subheader(
            "Uploaded Dataset Preview"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # --------------------------
        # Load Model
        # --------------------------

        model = joblib.load(
            "models/random_forest.pkl"
        )

        scaler = joblib.load(
            "models/scaler.pkl"
        )

        # --------------------------
        # Validation
        # --------------------------

        if "Class" in df.columns:
            df = df.drop(
                "Class",
                axis=1
            )

        # --------------------------
        # Scaling
        # --------------------------

        scaled_data = scaler.transform(
            df
        )

        # --------------------------
        # Prediction
        # --------------------------

        probabilities = (
            model.predict_proba(
                scaled_data
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        result_df = df.copy()

        result_df[
            "Fraud_Probability"
        ] = probabilities

        result_df[
            "Prediction"
        ] = predictions

        # --------------------------
        # Statistics
        # --------------------------

        fraud_count = (
            result_df[
                "Prediction"
            ] == 1
        ).sum()

        genuine_count = (
            result_df[
                "Prediction"
            ] == 0
        ).sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Fraud Transactions",
            fraud_count
        )

        col2.metric(
            "Genuine Transactions",
            genuine_count
        )

        # --------------------------
        # Probability Distribution
        # --------------------------

        st.markdown("---")

        st.subheader(
            "Fraud Probability Distribution"
        )

        st.plotly_chart(
            probability_distribution(
                probabilities
            ),
            use_container_width=True
        )

        # --------------------------
        # Results
        # --------------------------

        st.markdown("---")

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            result_df.head(20),
            use_container_width=True
        )

        # --------------------------
        # Download
        # --------------------------

        csv = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )