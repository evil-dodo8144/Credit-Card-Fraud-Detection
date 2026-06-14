import streamlit as st
import pandas as pd

from utils.plots import (
    amount_distribution_chart,
    fraud_correlation_chart,
    fraud_amount_boxplot
)

st.title("📈 Advanced Analytics")


@st.cache_data
def load_data():

    url = "https://drive.google.com/file/d/18vFxh2HBkXdEG5D9I4PmQ2YgPYvHNG75/view?usp=sharing"

    return pd.read_csv(url)

df =  load_data()

# -----------------------------------
# Sidebar Filters
# -----------------------------------

st.sidebar.header(
    "Analytics Filters"
)

max_amount = int(
    df["Amount"].max()
)

amount_filter = st.sidebar.slider(
    "Maximum Transaction Amount",
    min_value=0,
    max_value=max_amount,
    value=min(
        5000,
        max_amount
    )
)

filtered_df = df[
    df["Amount"] <= amount_filter
]

# -----------------------------------
# Summary
# -----------------------------------

st.subheader(
    "Filtered Dataset Summary"
)

st.write(
    f"Transactions Selected: "
    f"{len(filtered_df):,}"
)

# -----------------------------------
# Amount Analysis
# -----------------------------------

st.markdown("---")

st.subheader(
    "Transaction Amount Analysis"
)

st.plotly_chart(
    amount_distribution_chart(
        filtered_df
    ),
    use_container_width=True
)

# -----------------------------------
# Fraud Amount Analysis
# -----------------------------------

st.markdown("---")

st.subheader(
    "Fraud Transaction Analysis"
)

st.plotly_chart(
    fraud_amount_boxplot(
        filtered_df
    ),
    use_container_width=True
)

# -----------------------------------
# Correlation Analysis
# -----------------------------------

st.markdown("---")

st.subheader(
    "Features Most Related To Fraud"
)

st.plotly_chart(
    fraud_correlation_chart(
        filtered_df
    ),
    use_container_width=True
)

# -----------------------------------
# Statistics
# -----------------------------------

st.markdown("---")

st.subheader(
    "Statistical Summary"
)

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)

# -----------------------------------
# Fraud Insights
# -----------------------------------

fraud_df = filtered_df[
    filtered_df["Class"] == 1
]

st.markdown("---")

st.subheader(
    "Fraud Insights"
)

if len(fraud_df) > 0:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud Transactions",
        len(fraud_df)
    )

    col2.metric(
        "Average Fraud Amount",
        f"{fraud_df['Amount'].mean():.2f}"
    )

    col3.metric(
        "Maximum Fraud Amount",
        f"{fraud_df['Amount'].max():.2f}"
    )

else:

    st.warning(
        "No fraud transactions found under current filter."
    )

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.caption(
    "Credit Card Fraud Detection | AIML Project"
)