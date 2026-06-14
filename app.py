import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
### Detect Fraudulent Credit Card Transactions using Machine Learning

This project uses **Machine Learning** techniques to identify fraudulent credit card transactions from a highly imbalanced dataset.

The application provides:

- 📊 Interactive Dashboard
- 📈 Data Analytics & Visualizations
- ⚖️ Model Comparison
- 🔍 Batch Fraud Prediction
- 🎯 Threshold-based Evaluation
- 📉 Confusion Matrix Analysis

---

### Dataset Information

- Total Transactions: **284,807**
- Fraud Transactions: **492**
- Fraud Rate: **0.172%**
- Features: **Time, V1-V28, Amount**
- Target Variable: **Class**

---

### Machine Learning Models

- Logistic Regression
- Random Forest Classifier

---

### Evaluation Metrics

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

### Navigation

Use the **sidebar** to access different modules:

- Dashboard
- Analytics
- Model Comparison
- Batch Prediction

---
""")

# --------------------------------------------------
# Project Highlights
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        **Data Preprocessing**
        
        ✔ Null Value Check
        
        ✔ Duplicate Removal
        
        ✔ Feature Scaling
        
        ✔ Data Validation
        """
    )

with col2:
    st.success(
        """
        **Imbalance Handling**
        
        ✔ SMOTE
        
        ✔ Stratified Split
        
        ✔ Balanced Training
        
        ✔ Improved Recall
        """
    )

with col3:
    st.warning(
        """
        **Deployment Features**
        
        ✔ Interactive Charts
        
        ✔ CSV Upload
        
        ✔ Model Comparison
        
        ✔ Download Predictions
        """
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Built using Python, Scikit-Learn, SMOTE, Plotly, and Streamlit"
)