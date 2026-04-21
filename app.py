import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer is likely to churn.")

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.header("Customer Input Details")

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
FrequentFlyer_Encoded = st.selectbox("Frequent Flyer", [0, 1], help="0 = No, 1 = Yes")
AnnualIncomeClass_Encoded = st.selectbox(
    "Annual Income Class",
    [0, 1, 2],
    help="Use the same encoding as in your notebook"
)
ServicesOpted = st.number_input("Services Opted", min_value=1, max_value=10, value=3)
AccountSyncedToSocialMedia_Encoded = st.selectbox(
    "Account Synced To Social Media",
    [0, 1],
    help="0 = No, 1 = Yes"
)
BookedHotelOrNot_Encoded = st.selectbox(
    "Booked Hotel Or Not",
    [0, 1],
    help="0 = No, 1 = Yes"
)

input_df = pd.DataFram
