import streamlit as st
import pandas as pd
import pickle
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Churn Predictor",
    layout="wide",
    page_icon="🤖"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Global background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 30px rgba(0,255,255,0.15);
    margin-bottom: 20px;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00f7ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Neon Button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00f7ff, #00ff88);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    border: none;
    box-shadow: 0 0 15px rgba(0,255,255,0.6);
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,255,1);
}

/* Progress bar */
.progress-bar {
    height: 20px;
    border-radius: 10px;
    background: #1e293b;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00f7ff, #00ff88);
    text-align: center;
    color: black;
    font-weight: bold;
}

/* Result box */
.result {
    text-align: center;
    font-size: 22px;
    padding: 20px;
    border-radius: 15px;
}

.success {
    background: rgba(0,255,150,0.15);
}

.error {
    background: rgba(255,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer behavior with Machine Learning</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Adjust inputs and predict churn risk.")

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])

with col2:
    annual_income = st.selectbox("Income Level", ["Low", "Medium", "High"])
    services_opted = st.slider("Services Opted", 1, 10, 3)

with col3:
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

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Now"):

    # Loading animation
    with st.spinner("Analyzing customer behavior..."):
        time.sleep(1.5)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    # Progress bar
    percent = int(probability * 100)
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width:{percent}%">
            {percent}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Result text
    if prediction == 1:
        st.markdown(f"""
        <div class="result error">
        ❌ High Risk of Churn <br>
        Probability: {probability:.2f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result success">
        ✅ Customer Retained <br>
        Probability: {probability:.2f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
