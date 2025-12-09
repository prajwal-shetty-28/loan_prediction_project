import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ------------------------------------
# Load the trained Loan Model
# ------------------------------------
try:
    model = joblib.load("loan_model.joblib")  # Use joblib, not pickle
except FileNotFoundError:
    st.error("Model file 'loan_model.joblib' not found. Make sure it is in the project folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# ------------------------------------
# Prediction function
# ------------------------------------
def predict_loan(Income, Loan_Amount, Credit_Score, Employment_Status, DTI_Ratio, Text):
    try:
        input_data = pd.DataFrame([{
            'Income': float(Income),
            'Loan_Amount': float(Loan_Amount),
            'Credit_Score': float(Credit_Score),
            'Employment_Status': Employment_Status,
            'DTI_Ratio': float(DTI_Ratio),
            'Text': Text
        }])

        prediction = model.predict(input_data)[0]

        if prediction in ['Y', 1, 'Approved']:
            return "Loan Approved ✔"
        else:
            return "Loan Rejected ❌"

    except Exception as e:
        return f"Error: {e}"

# ------------------------------------
# Main Streamlit App
# ------------------------------------
def main():
    st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
    st.title("Loan Approval Prediction App")

    st.markdown("""
    <div style="background-color:purple;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Approval Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # Inputs
    # -------------------------------
    Income = st.number_input("Income", min_value=0.0, step=1000.0)
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
    Credit_Score = st.number_input("Credit Score", min_value=0.0, max_value=1000.0, step=1.0)
    Employment_Status = st.selectbox(
        "Employment Status",
        ("Salaried", "Self-Employed", "Unemployed", "Student", "Retired")
    )
    DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, step=0.01)
    Text = st.text_input("Additional Notes", "")

    result = ""

    # -------------------------------
    # Predict Button
    # -------------------------------
    if st.button("Predict"):
        result = predict_loan(
            Income, Loan_Amount, Credit_Score,
            Employment_Status, DTI_Ratio, Text
        )
        st.success(f"The result is: {result}")

    # -------------------------------
    # About Section
    # -------------------------------
    st.markdown("---")
    if st.button("About"):
        st.info("Loan Approval Prediction App built with Streamlit and a trained ML model.")
        st.text("Enter applicant details above and click Predict to see the result.")

if __name__ == '__main__':
    main()
