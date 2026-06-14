import os
import json
import joblib
import warnings
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Create Required Directories
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading Dataset...")

from data_loader import load_data

df = load_data()

print(f"\nOriginal Dataset Shape: {df.shape}")

# --------------------------------------------------
# Null Value Check
# --------------------------------------------------

print("\nChecking for Null Values...")

null_values = df.isnull().sum()

print(null_values)

if null_values.sum() > 0:
    df = df.dropna()
    print("\nNull values removed.")

# --------------------------------------------------
# Duplicate Value Check
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate Rows Found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

print(f"\nCleaned Dataset Shape: {df.shape}")

# --------------------------------------------------
# Data Types
# --------------------------------------------------

print("\nDataset Information")

print(df.info())

# --------------------------------------------------
# Class Distribution
# --------------------------------------------------

print("\nClass Distribution")

print(df["Class"].value_counts())

fraud_percentage = (
    df["Class"].value_counts()[1] /
    len(df)
) * 100

print(
    f"\nFraud Percentage: "
    f"{fraud_percentage:.4f}%"
)

# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# SMOTE on Training Data Only
# --------------------------------------------------

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE")

print(pd.Series(y_train_smote).value_counts())

# --------------------------------------------------
# Model Initialization
# --------------------------------------------------

models = {
    "Logistic Regression":
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

metrics = {}

# --------------------------------------------------
# Train Models
# --------------------------------------------------

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(
        X_train_smote,
        y_train_smote
    )

    y_pred = model.predict(
        X_test_scaled
    )

    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    metrics[model_name] = {
        "accuracy":
            round(accuracy, 4),

        "precision":
            round(precision, 4),

        "recall":
            round(recall, 4),

        "f1_score":
            round(f1, 4),

        "roc_auc":
            round(roc_auc, 4),

        "confusion_matrix":
            cm.tolist()
    }

    print("\nModel Performance")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

# --------------------------------------------------
# Save Models
# --------------------------------------------------

print("\nSaving Models...")

joblib.dump(
    models["Logistic Regression"],
    "models/logistic_regression.pkl"
)

joblib.dump(
    models["Random Forest"],
    "models/random_forest.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# --------------------------------------------------
# Save Metrics Report
# --------------------------------------------------

with open(
    "reports/metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("\nMetrics Saved")

# --------------------------------------------------
# Training Summary
# --------------------------------------------------

print("\nTraining Completed Successfully")

print(
    "\nSaved Files:"
)

print(
    "- models/logistic_regression.pkl"
)

print(
    "- models/random_forest.pkl"
)

print(
    "- models/scaler.pkl"
)

print(
    "- reports/metrics.json"
)