# ==========================================
# Loan Approval Prediction App (Streamlit)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tempfile
import io
import base64

# --- Load model and dataset ---
@st.cache_resource
def load_model_and_data():
    model = joblib.load("loan_model.joblib")
    df = pd.read_csv("loan_acceptance_dataset.csv")
    X = pd.get_dummies(df.drop(['applicant_id', 'loan_approved'], axis=1), drop_first=True)
    return model, X.columns.tolist()

model, feature_names = load_model_and_data()

# --- Confidence message helper ---
def confidence_message(pred, prob):
    if pred == 1 and prob > 0.95:
        return "The model is VERY confident that this loan will be APPROVED."
    elif pred == 1:
        return "The model predicts APPROVAL, but with moderate confidence."
    elif pred == 0 and prob < 0.05:
        return "The model is VERY confident that this loan will be REJECTED."
    else:
        return "The model predicts REJECTION, but with moderate confidence."

# --- Sidebar Navigation ---
st.sidebar.title("Loan Prediction App")
page = st.sidebar.radio(
    "Select a Feature:",
    ["Single Loan Prediction", "Batch Predictions", "Feature Importance"]
)

# =====================================
# 1. SINGLE LOAN PREDICTION
# =====================================
if page == "Single Loan Prediction":
    st.title("Loan Approval Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Applicant Age", 18, 70, 30)
        income = st.number_input("Annual Income (USD)", min_value=0)
        employment_years = st.slider("Years of Employment", 0, 40, 5)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        loan_amount = st.number_input("Loan Amount Requested (USD)", min_value=1000)
    with col2:
        loan_term_months = st.slider("Loan Term (Months)", 6, 60, 24, step=6)
        existing_loans_count = st.slider("Existing Loans", 0, 5, 1)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Personal"])

    if st.button("Predict Loan Outcome"):
        # Prepare input
        input_dict = {
            'age': [age],
            'income': [income],
            'employment_years': [employment_years],
            'credit_score': [credit_score],
            'loan_amount': [loan_amount],
            'loan_term_months': [loan_term_months],
            'existing_loans_count': [existing_loans_count],
            'marital_status': [marital_status],
            'education_level': [education_level],
            'loan_purpose': [loan_purpose]
        }

        input_df = pd.DataFrame(input_dict)
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        for col in set(feature_names) - set(input_encoded.columns):
            input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]

        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        result = "Loan Approved" if pred == 1 else "Loan Rejected"
        st.subheader(result)
        st.write(f"Approval Probability: {prob:.2%}")
        st.info(confidence_message(pred, prob))

# =====================================
# 2. BATCH PREDICTIONS
# =====================================
elif page == "Batch Predictions":
    st.title("Batch Loan Predictions")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                new_data = pd.read_csv(uploaded_file)
            else:
                new_data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        # Encode and align
        new_data_encoded = pd.get_dummies(new_data, drop_first=True)
        for col in set(feature_names) - set(new_data_encoded.columns):
            new_data_encoded[col] = 0
        new_data_encoded = new_data_encoded[feature_names]

        pred_labels = model.predict(new_data_encoded)
        pred_probs = model.predict_proba(new_data_encoded)[:, 1]

        results_df = new_data.copy()
        results_df["Prediction"] = np.where(pred_labels == 1, "Loan Approved", "Loan Rejected")
        results_df["Approval_Probability"] = pred_probs
        results_df["Confidence_Message"] = [
            confidence_message(pred, prob) for pred, prob in zip(pred_labels, pred_probs)
        ]

        st.subheader("Preview of Results (first 5 rows)")
        st.dataframe(results_df.head())

        # Download link
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv_data).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="loan_predictions.csv">Download Full Results as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# =====================================
# 3. FEATURE IMPORTANCE
# =====================================
elif page == "Feature Importance":
    st.title("Model Feature Importance")

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    st.write("The bar chart below shows which features most strongly impact loan approval predictions:")

    fig, ax = plt.subplots(figsize=(8, 6))
    top_features = coef_df.head(10)
    ax.barh(top_features["Feature"], top_features["Coefficient"], color="skyblue")
    ax.set_xlabel("Coefficient Value (Impact Strength)")
    ax.set_title("Top 10 Important Features (Logistic Regression)")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

    st.dataframe(coef_df.head(15))
