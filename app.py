import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Churn Predictor",
    layout="centered",
    page_icon="🤖"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Glass container */
.main {
    background: rgba(255, 255, 255, 0.05);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 0 40px rgba(0,255,255,0.2);
}

/* Title */
h1 {
    text-align: center;
    color: #00f7ff;
    text-shadow: 0 0 20px #00f7ff;
}

/* Inputs */
.stNumberInput, .stSelectbox {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #00f7ff, #00ff88);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    box-shadow: 0 0 20px rgba(0,255,255,0.6);
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px rgba(0,255,255,1);
}

/* Success & Error */
.stAlert {
    border-radius: 15px;
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🤖 AI Customer Churn Predictor")
st.markdown("### Predict customer behavior using Machine Learning")

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- FORM ----------------
st.markdown("## 🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
    annual_income = st.selectbox("Annual Income", ["Low", "Medium", "High"])

with col2:
    services_opted = st.number_input("Services Opted", min_value=1, max_value=10, value=3)
    social_media = st.selectbox("Social Media Linked", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

# ---------------- DATA ----------------
input_df = pd.DataFrame([{
    "Age": age,
    "FrequentFlyer_Encoded": 1 if frequent_flyer == "Yes" else 0,
    "AnnualIncomeClass_Encoded": {"Low": 0, "Medium": 1, "High": 2}[annual_income],
    "ServicesOpted": services_opted,
    "AccountSyncedToSocialMedia_Encoded": 1 if social_media == "Yes" else 0,
    "BookedHotelOrNot_Encoded": 1 if booked_hotel == "Yes" else 0
}])

# ---------------- PREDICTION ----------------
st.markdown("")

if st.button("🚀 Predict Churn"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div style='background: rgba(255,0,0,0.2); padding:20px; border-radius:15px;'>
                ❌ <b>Customer WILL churn</b><br>
                Probability: <b>{probability:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style='background: rgba(0,255,150,0.2); padding:20px; border-radius:15px;'>
                ✅ <b>Customer will NOT churn</b><br>
                Probability: <b>{probability:.2f}</b>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
