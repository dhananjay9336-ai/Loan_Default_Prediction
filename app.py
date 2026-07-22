# import streamlit as st
# import pandas as pd
# import joblib

# # Load Model
# model = joblib.load("saved_model/loan_model.pkl")
# scaler = joblib.load("saved_model/scaler.pkl")
# encoders = joblib.load("saved_model/encoders.pkl")

# # Title
# st.title("🏦 Loan Default Prediction System")

# st.write("Enter customer details below")



import streamlit as st
import pandas as pd
import joblib

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==============================
# Load Model
# ==============================

model = joblib.load("saved_model/loan_model.pkl")
scaler = joblib.load("saved_model/scaler.pkl")
encoders = joblib.load("saved_model/encoders.pkl")



st.title("🏦 Loan Default Prediction System")

st.markdown("---")

st.write("""
This application predicts whether a customer is likely to repay a loan
based on financial and personal information.
""")

col1, col2 = st.columns(2)
with col1:

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0
    )
    

    debt_to_income_ratio = st.number_input(
        "Debt To Income Ratio",
        min_value=0.0,
        value=0.20
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )


with col2:

    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=10.5
    )

    gender = st.selectbox(
        "Gender",
        encoders["gender"].classes_
    )

    marital_status = st.selectbox(
        "Marital Status",
        encoders["marital_status"].classes_
    )

    education_level = st.selectbox(
        "Education Level",
        encoders["education_level"].classes_
    )

    employment_status = st.selectbox(
    "Employment Status",
    encoders["employment_status"].classes_
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    encoders["loan_purpose"].classes_
)

grade_subgrade = st.selectbox(
    "Grade Subgrade",
    encoders["grade_subgrade"].classes_
)



# ==============================
# Prediction Button
# ==============================

if st.button("Predict Loan Status"):
    gender = encoders["gender"].transform([gender])[0]

    marital_status = encoders["marital_status"].transform([marital_status])[0]

    education_level = encoders["education_level"].transform([education_level])[0]

    employment_status = encoders["employment_status"].transform([employment_status])[0]

    loan_purpose = encoders["loan_purpose"].transform([loan_purpose])[0]

    grade_subgrade = encoders["grade_subgrade"].transform([grade_subgrade])[0]

    input_data = pd.DataFrame({

    "annual_income":[annual_income],

    "debt_to_income_ratio":[debt_to_income_ratio],

    "credit_score":[credit_score],

    "loan_amount":[loan_amount],

    "interest_rate":[interest_rate],

    "gender":[gender],

    "marital_status":[marital_status],

    "education_level":[education_level],

    "employment_status":[employment_status],

    "loan_purpose":[loan_purpose],

    "grade_subgrade":[grade_subgrade]

    })
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    st.markdown("---")

    if prediction[0] == 1:

        st.success("✅ Customer is likely to PAY BACK the loan.")

    else:
        st.error("❌ Customer is likely to DEFAULT on the loan.")
        st.subheader("Prediction Probability")
        
        st.write(f"Default Probability : {probability[0][0]*100:.2f}%")
        
        st.write(f"Paid Back Probability : {probability[0][1]*100:.2f}%")


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# CSS
def load_css():
    with open("styles/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# Load Model
model = joblib.load("saved_model/loan_model.pkl")