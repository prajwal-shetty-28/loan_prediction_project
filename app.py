import numpy as np
import pickle
import pandas as pd
import streamlit as st

# ------------------------------------
# Load the trained Loan Model
# ------------------------------------
pickle_in = open("loan_data.pkl", "rb")
model = pickle.load(pickle_in)

def predict_loan(Income, Loan_Amount, Credit_Score, Employment_Status, DTI_Ratio, Text):
    try:
        # Create dataframe with the exact same columns used during training
        input_data = pd.DataFrame([{
            'Income': float(Income),
            'Loan_Amount': float(Loan_Amount),
            'Credit_Score': float(Credit_Score),
            'Employment_Status': Employment_Status,
            'DTI_Ratio': float(DTI_Ratio),
            'Text': Text
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        # Convert prediction to text label
        if prediction in ['Y', 1, 'Approved']:
            return "Loan Approved ✔"
        else:
            return "Loan Rejected ❌"

    except Exception as e:
        return f"Error: {e}"


# ------------------------------------
# MAIN STREAMLIT APP
# ------------------------------------
def main():
    st.title("Loan Approval Prediction App")

    html_temp = """
    <div style="background-color:purple;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Approval Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # -------------------------------
    # INPUT FIELDS (MATCHING MODEL)
    # -------------------------------
    Income = st.text_input("Income", "Enter applicant income")

    Loan_Amount = st.text_input("Loan Amount", "Enter loan amount")

    Credit_Score = st.text_input("Credit Score", "Enter credit score")

    Employment_Status = st.selectbox(
        "Employment Status",
        ("Salaried", "Self-Employed", "Unemployed", "Student", "Retired")
    )

    DTI_Ratio = st.text_input("DTI Ratio", "Enter debt-to-income ratio (0-1)")

    Text = st.text_input("Text", "Enter any description / notes")

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
    if st.button("About"):
        st.text("Loan Approval Prediction Model")
        st.text("Built using Streamlit + ML Pipeline")


if __name__ == '__main__':
    main()
