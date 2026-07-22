# ============================================
# Loan Default Prediction System
# Author : Dhananjay Yadav
# ============================================

# ========== Import Libraries ==========

import streamlit as st
import pandas as pd
import joblib
import os

# ========== Page Configuration ==========

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Load Custom CSS
# ============================================

def load_css():
    """
    Load custom CSS file for professional UI.
    """
    try:
        with open("styles/style.css", "r") as css:
            st.markdown(
                f"<style>{css.read()}</style>",
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning("style.css file not found.")

# Load CSS
load_css()


# ============================================
# Load Trained Model & Saved Files
# ============================================

try:
    model = joblib.load("saved_model/loan_model.pkl")
    scaler = joblib.load("saved_model/scaler.pkl")
    encoders = joblib.load("saved_model/encoders.pkl")

except Exception as e:
    st.error(f"Error loading model files:\n{e}")
    st.stop()


# # ============================================
# # Sidebar Debug Information
# # ============================================

# st.sidebar.title("⚙ Debug Information")

# # Model Path
# st.sidebar.subheader("📂 Model Path")
# st.sidebar.code(os.path.abspath("saved_model/loan_model.pkl"))

# # Model Type
# st.sidebar.subheader("🤖 Model Type")
# st.sidebar.write(type(model))

# # Loaded Model
# st.sidebar.subheader("📌 Loaded Model")
# st.sidebar.write(model)

# # Scaler Type
# st.sidebar.subheader("📏 Scaler")
# st.sidebar.write(type(scaler))

# # Available Encoders
# st.sidebar.subheader("🏷 Encoders")
# st.sidebar.write(list(encoders.keys()))

# ============================================
# Main Title
# ============================================

st.title("🏦 Loan Default Prediction System")

st.markdown(
    """
    ### 🤖 Machine Learning Based Loan Risk Analysis

    Welcome to the **Loan Default Prediction System**.

    This application uses a trained **Machine Learning Model**
    to predict whether a customer is likely to **repay** or
    **default** on a loan based on financial and personal details.

    ---
    """
)

# ============================================
# Information Cards
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        ### 📊 Input

        Enter the customer's
        financial and personal
        information.
        """
    )

with col2:
    st.warning(
        """
        ### 🤖 Prediction

        The trained ML model
        analyzes the data
        and predicts loan status.
        """
    )

with col3:
    st.success(
        """
        ### 📈 Result

        View prediction along
        with confidence
        probabilities.
        """
    )

st.markdown("---")

# ============================================
# Customer Information Form
# ============================================

st.header("📝 Customer Information")

st.write("Please enter the customer's financial and personal details below.")

st.markdown("---")

# Create Two Columns
col1, col2 = st.columns(2)

# ============================================
# Left Column
# ============================================

with col1:

    annual_income = st.number_input(
        "💰 Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        help="Enter customer's annual income."
    )

    debt_to_income_ratio = st.number_input(
        "📊 Debt to Income Ratio",
        min_value=0.0,
        max_value=5.0,
        value=0.20,
        step=0.01,
        help="Example: 0.25 means 25%."
    )

    credit_score = st.number_input(
        "📈 Credit Score",
        min_value=300,
        max_value=850,
        value=700,
        step=1,
        help="Valid range: 300 - 850"
    )

    loan_amount = st.number_input(
        "💵 Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        help="Requested loan amount."
    )

    interest_rate = st.number_input(
        "📉 Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.5,
        step=0.1,
        help="Annual interest rate."
    )

# ============================================
# Right Column
# ============================================

with col2:

    gender = st.selectbox(
        "👤 Gender",
        encoders["gender"].classes_,
        help="Select customer gender."
    )

    marital_status = st.selectbox(
        "💍 Marital Status",
        encoders["marital_status"].classes_,
        help="Select marital status."
    )

    education_level = st.selectbox(
        "🎓 Education Level",
        encoders["education_level"].classes_,
        help="Select education level."
    )

    employment_status = st.selectbox(
        "💼 Employment Status",
        encoders["employment_status"].classes_,
        help="Select employment status."
    )

    loan_purpose = st.selectbox(
        "🏦 Loan Purpose",
        encoders["loan_purpose"].classes_,
        help="Purpose of loan."
    )

    grade_subgrade = st.selectbox(
        "⭐ Grade / Subgrade",
        encoders["grade_subgrade"].classes_,
        help="Select loan grade."
    )

st.markdown("---")

# Prediction Button
predict_btn = st.button(
    "🔍 Predict Loan Status",
    use_container_width=True
)

# ============================================
# Prediction Logic
# ============================================

if predict_btn:

    # ----------------------------------------
    # Encode Categorical Features
    # ----------------------------------------

    gender_encoded = encoders["gender"].transform([gender])[0]

    marital_status_encoded = encoders["marital_status"].transform(
        [marital_status]
    )[0]

    education_level_encoded = encoders["education_level"].transform(
        [education_level]
    )[0]

    employment_status_encoded = encoders["employment_status"].transform(
        [employment_status]
    )[0]

    loan_purpose_encoded = encoders["loan_purpose"].transform(
        [loan_purpose]
    )[0]

    grade_subgrade_encoded = encoders["grade_subgrade"].transform(
        [grade_subgrade]
    )[0]

    # ----------------------------------------
    # Create Input DataFrame
    # ----------------------------------------

    input_data = pd.DataFrame({

        "annual_income": [annual_income],
        "debt_to_income_ratio": [debt_to_income_ratio],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "interest_rate": [interest_rate],
        "gender": [gender_encoded],
        "marital_status": [marital_status_encoded],
        "education_level": [education_level_encoded],
        "employment_status": [employment_status_encoded],
        "loan_purpose": [loan_purpose_encoded],
        "grade_subgrade": [grade_subgrade_encoded]

    })

    # ----------------------------------------
    # # Feature Order Check
    # # ----------------------------------------

    # st.sidebar.subheader("📋 Feature Order")

    # st.sidebar.write(input_data.columns.tolist())

    # ----------------------------------------
    # Scale Input
    # ----------------------------------------

    input_scaled = scaler.transform(input_data)

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    # # ----------------------------------------
    # # Prediction Details
    # # ----------------------------------------

    # st.markdown("---")

    # st.subheader("🔍 Prediction Details")

    # st.write("Prediction :", prediction[0])

    # st.write("Probability Array")

    # st.write(probability)

    # st.write(
    #     "Default Probability :",
    #     float(probability[0][0])
    # )

    # st.write(
    #     "Paid Back Probability :",
    #     float(probability[0][1])
    # )

    st.markdown("---")

    # # ============================================
    # # Debug Output
    # # ============================================

    # st.header("🛠 Debug Information")

    # # Encoded Input
    # st.subheader("📋 Encoded Input Data")

    # st.dataframe(
    #     input_data,
    #     use_container_width=True
    # )

    # # Scaled Input
    # st.subheader("📊 Scaled Input Data")

    # scaled_df = pd.DataFrame(
    #     input_scaled,
    #     columns=input_data.columns
    # )

    # st.dataframe(
    #     scaled_df,
    #     use_container_width=True
    # )

    # st.markdown("---")

    # ============================================
    # Final Prediction Result
    # ============================================

    st.header("🎯 Prediction Result")

    default_probability = float(probability[0][0])
    paid_probability = float(probability[0][1])

    confidence = max(default_probability, paid_probability) * 100

    if prediction[0] == 1:

        st.success(
            "✅ Customer is likely to PAY BACK the loan."
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    else:

        st.error(
            "❌ Customer is likely to DEFAULT on the loan."
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )
    st.markdown("---")

    # ============================================
    # Probability Details
    # ============================================

    st.subheader("📈 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Default Probability",
            value=f"{default_probability*100:.2f}%"
        )

    with col2:

        st.metric(
            label="Paid Back Probability",
            value=f"{paid_probability*100:.2f}%"
        )

    st.markdown("---")

    # # ============================================
    # # Raw Model Output
    # # ============================================

    # with st.expander("🔍 View Raw Model Output"):

    #     st.write("Prediction :", prediction)

    #     st.write("Probability Array :")

    #     st.write(probability)

    #     st.write("Feature Order :")

    #      st.write(input_data.columns.tolist())

    #     st.write("Input Shape :", input_data.shape)

    #     st.write("Scaled Shape :", input_scaled.shape)