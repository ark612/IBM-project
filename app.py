import streamlit as st
import pandas as pd
import pickle
import time

st.set_page_config(page_title="AI Churn Predictor", layout="wide")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #020617, #000);
    color: white;
}
.title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(15, 23, 42, 0.6);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(56,189,248,0.2);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Customer Churn Dashboard</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Insights", "📜 History"])

# ================= TAB 1 =================
with tab1:
    st.markdown("### Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
        annual_income = st.selectbox("Income Level", ["Low", "Medium", "High"])

    with col2:
        services_opted = st.slider("Services Opted", 1, 10, 3)
        social_media = st.selectbox("Social Media Linked", ["No", "Yes"])
        booked_hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

    input_df = pd.DataFrame([{
        "Age": age,
        "FrequentFlyer_Encoded": 1 if frequent_flyer == "Yes" else 0,
        "AnnualIncomeClass_Encoded": {"Low": 0, "Medium": 1, "High": 2}[annual_income],
        "ServicesOpted": services_opted,
        "AccountSyncedToSocialMedia_Encoded": 1 if social_media == "Yes" else 0,
        "BookedHotelOrNot_Encoded": 1 if booked_hotel == "Yes" else 0
    }])

    if st.button("🚀 Predict"):

        with st.spinner("Running AI model..."):
            time.sleep(1)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        percent = int(probability * 100)

        st.markdown("### 📊 Result")
        st.progress(percent / 100)

        if prediction == 1:
            st.error(f"High Risk of Churn ({percent}%)")
        else:
            st.success(f"Low Risk ({percent}%)")

        # Save history
        st.session_state.history.append({
            "Age": age,
            "Income": annual_income,
            "Services": services_opted,
            "Risk %": percent
        })

# ================= TAB 2 =================
with tab2:
    st.markdown("### 🧠 AI Insights")

    st.markdown("#### Why this prediction?")

    reasons = []

    if services_opted < 3:
        reasons.append("Low engagement (few services used)")
    if frequent_flyer == "No":
        reasons.append("Not a frequent flyer")
    if annual_income == "Low":
        reasons.append("Lower income segment")
    if booked_hotel == "No":
        reasons.append("No hotel bookings")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("Customer shows strong engagement.")

    st.markdown("---")

    st.markdown("#### What-if Analysis")

    test_services = st.slider("Change Services Opted", 1, 10, 5)

    test_df = input_df.copy()
    test_df["ServicesOpted"] = test_services

    new_prob = model.predict_proba(test_df)[0][1]
    st.write(f"New churn probability: {new_prob:.2f}")

# ================= TAB 3 =================
with tab3:
    st.markdown("### 📜 Prediction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.write("No predictions yet.")
