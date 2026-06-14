import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


def fraud_distribution_chart(df):

    counts = df["Class"].value_counts()

    fig = px.pie(
        values=counts.values,
        names=["Genuine", "Fraud"],
        title="Fraud vs Genuine Transactions"
    )

    return fig


def amount_distribution_chart(df):

    fig = px.histogram(
        df,
        x="Amount",
        color="Class",
        nbins=100,
        title="Transaction Amount Distribution"
    )

    return fig


def fraud_correlation_chart(df):

    corr = (
        df.corr(numeric_only=True)["Class"]
        .abs()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        x=corr.index,
        y=corr.values,
        title="Top Features Correlated With Fraud"
    )

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Absolute Correlation"
    )

    return fig


def fraud_amount_boxplot(df):

    fraud_df = df[
        df["Class"] == 1
    ]

    fig = px.box(
        fraud_df,
        y="Amount",
        title="Fraud Transaction Amount Distribution"
    )

    return fig


def probability_distribution(probabilities):

    fig = px.histogram(
        x=probabilities,
        nbins=50,
        title="Fraud Probability Distribution"
    )

    return fig


def confusion_matrix_plot(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        title="Confusion Matrix"
    )

    return fig


def roc_curve_plot(
    y_true,
    y_prob
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.4f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random"
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    return fig