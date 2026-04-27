import streamlit as st
import pandas as pd
import pickle
import time

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #020617, #000);
    color: #e2e8f0;
}

/* Title */
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: rgba(15, 23, 42, 0.6);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(56,189,248,0.2);
    box-shadow: 0 0 20px rgba(56,189,248,0.1);
    margin-bottom: 20px;
}

/* Inputs */
.stNumberInput input, .stSelectbox {
    background-color: #020617 !important;
    color: white !important;
    border: 1px solid #0ea5e9 !important;
}

/* Button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    border: none;
    box-shadow: 0 0 15px rgba(56,189,248,0.6);
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
}

/* Result */
.result {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

/* Progress bar */
.progress-container {
    background: #020617;
    border-radius: 10px;
    height: 15px;
    margin-top: 10px;
}

.progress-bar {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">AI Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-powered prediction • real-time analysis</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- INPUT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
    annual_income = st.selectbox("Annual Income", ["Low", "Medium", "High"])

with col2:
    services_opted = st.number_input("Services Opted", min_value=1, max_value=10, value=3)
    social_media = st.selectbox("Social Media Linked", ["No", "Yes"])
    booked_hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DATA ----------------
input_df = pd.DataFrame([{
    "Age": age,
    "FrequentFlyer_Encoded": 1 if frequent_flyer == "Yes" else 0,
    "AnnualIncomeClass_Encoded": {"Low": 0, "Medium": 1, "High": 2}[annual_income],
    "ServicesOpted": services_opted,
    "AccountSyncedToSocialMedia_Encoded": 1 if social_media == "Yes" else 0,
    "BookedHotelOrNot_Encoded": 1 if booked_hotel == "Yes" else 0
}])

# ---------------- PREDICT ----------------
if st.button("Analyze Customer"):

    with st.spinner("Running AI model..."):
        time.sleep(1)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    percent = int(probability * 100)

    st.markdown('<div class="card result">', unsafe_allow_html=True)

    st.markdown(f"<h1 style='color:#38bdf8'>{percent}%</h1>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width:{percent}%"></div>
    </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.write("⚠️ High Risk of Churn")
    else:
        st.write("✅ Customer Likely to Stay")

    st.markdown('</div>', unsafe_allow_html=True)
