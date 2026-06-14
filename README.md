# 💳 Credit Card Fraud Detection System

## 📌 Overview

This project detects fraudulent credit card transactions using Machine Learning on a highly imbalanced real-world dataset. Since fraud cases account for only **0.172%** of transactions, SMOTE is used to balance the training data and improve fraud detection performance.

The project is deployed using **Streamlit** and provides interactive visualizations, model comparison, and batch fraud prediction.

---

## 🎯 Goal

* Detect fraudulent transactions accurately.
* Handle class imbalance using SMOTE.
* Compare machine learning models.
* Visualize fraud-related insights.
* Deploy a user-friendly web application.

---

## 🛠️ Technologies Used

**Programming & Data Processing**

* Python
* Pandas
* NumPy

**Machine Learning**

* Scikit-Learn
* Imbalanced-Learn (SMOTE)

**Models**

* Logistic Regression
* Random Forest Classifier

**Visualization**

* Plotly
* Matplotlib
* Seaborn

**Deployment**

* Streamlit

**Model Persistence**

* Joblib

---

## 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── requirements.txt
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── reports/
│   └── metrics.json
│
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Analytics.py
│   ├── 3_Model_Comparison.py
│   └── 4_Batch_Prediction.py
│
├── utils/
│   ├── train.py
│   └── plots.py
│
└── README.md
```

---

## ✨ Features Implemented

* Null Value Checking
* Duplicate Removal
* Feature Scaling
* SMOTE Oversampling
* Logistic Regression Model
* Random Forest Model
* Interactive Dashboard
* Fraud Distribution Analysis
* Correlation Heatmap
* Threshold-Based Prediction
* ROC Curve Visualization
* Confusion Matrix
* Batch CSV Prediction
* Downloadable Results

---

## ⚙️ Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
SMOTE (Training Data Only)
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Streamlit Deployment
```

---

## 🚀 Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python utils/train.py
```

### Launch Application

```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

## 🔮 Future Enhancements

* SHAP Explainability
* XGBoost / LightGBM Models
* Precision-Recall Curve
* Real-Time Fraud Detection API
* User Authentication
* Cloud Deployment (AWS, Azure, GCP)

---

## 📖 Conclusion

This project demonstrates an end-to-end machine learning pipeline for credit card fraud detection, including data preprocessing, imbalance handling, model training, evaluation, visualization, and deployment through an interactive Streamlit application.
