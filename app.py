import streamlit as st
import pandas as pd
import pickle
import time

st.set_page_config(page_title="AI Customer Predictor", layout="wide")

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
    font-size: 40px;
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
.result {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Customer Prediction Dashboard</div>', unsafe_allow_html=True)
st.write("### Predict whether a customer will repeat or not")

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔮 Predict", "📜 History"])

# ================= TAB 1 =================
with tab1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        frequent_flyer = st.selectbox("Frequent Flyer", ["No", "Yes"])
        annual_income = st.selectbox("Income Level", ["Low", "Medium", "High"])

    with col2:
        services_opted = st.slider("Services Opted", 1, 10, 3)
        social_media = st.selectbox("Social Media Linked", ["No", "Yes"])
        booked_hotel = st.selectbox("Booked Hotel", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

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

        repeat_prob = 1 - probability
        non_repeat_prob = probability

        st.markdown('<div class="card result">', unsafe_allow_html=True)

        if prediction == 1:
            st.error("❌ Customer will NOT repeat")
        else:
            st.success("✅ Customer will REPEAT")

        st.write("### 📊 Prediction Confidence")

        st.write(f"Repeat Probability: {repeat_prob:.2f}")
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-bar" style="width:{int(repeat_prob*100)}%"></div>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"Non-Repeat Probability: {non_repeat_prob:.2f}")
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-bar" style="width:{int(non_repeat_prob*100)}%"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Save history
        st.session_state.history.append({
            "Age": age,
            "Income": annual_income,
            "Services": services_opted,
            "Prediction": "Repeat" if prediction == 0 else "Not Repeat",
            "Repeat Prob": round(repeat_prob, 2),
            "Non-Repeat Prob": round(non_repeat_prob, 2)
        })

# ================= TAB 2 =================
with tab2:

    st.markdown("### 📜 Prediction History")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df, use_container_width=True)

        # ---------------- GRAPHS ----------------
        st.markdown("### 📊 Analytics")

        st.subheader("📈 Repeat Probability Trend")
        st.line_chart(df["Repeat Prob"])

        st.subheader("📊 Prediction Distribution")
        st.bar_chart(df["Prediction"].value_counts())

        st.subheader("🔍 Services vs Repeat Probability")
        st.scatter_chart(df[["Services", "Repeat Prob"]])

        # ---------------- DELETE OPTIONS ----------------
        st.markdown("### 🗑️ Manage History")

        index_to_delete = st.number_input(
            "Enter row index to delete",
            min_value=0,
            max_value=len(df)-1,
            step=1
        )

        if st.button("❌ Delete Selected Row"):
            st.session_state.history.pop(index_to_delete)
            st.rerun()

        if st.button("🗑️ Clear All History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.write("No predictions yet.")
