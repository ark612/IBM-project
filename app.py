import streamlit as st
import time

st.set_page_config(page_title="Attendance Risk Predictor", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #020617, #020617 40%, #000000);
    color: #e2e8f0;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 25px;
}

/* Glass card */
.card {
    background: rgba(15, 23, 42, 0.6);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(56,189,248,0.2);
    box-shadow: 0 0 20px rgba(56,189,248,0.1);
    margin-bottom: 20px;
}

/* Inputs */
input, .stNumberInput input {
    background-color: #020617 !important;
    color: white !important;
    border: 1px solid #0ea5e9 !important;
}

/* Button */
div.stButton > button {
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

/* Stats boxes */
.stat-box {
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    background: rgba(2,6,23,0.8);
    border: 1px solid rgba(56,189,248,0.3);
}

.stat-number {
    font-size: 26px;
    color: #38bdf8;
    font-weight: bold;
}

/* Result card */
.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.warning {
    background: rgba(59,130,246,0.15);
    border: 1px solid #3b82f6;
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
st.markdown('<div class="title">Attendance Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML-powered academic analytics • real-time projection</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

total = st.number_input("Total Classes Conducted", value=80)
attended = st.number_input("Classes Attended", value=50)
remaining = st.number_input("Remaining Classes", value=20)
planned_leave = st.number_input("Planned Absences", value=0)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- STATS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{attended}</div>Classes Attended</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{remaining}</div>Remaining</div>', unsafe_allow_html=True)

with col3:
    will_attend = remaining - planned_leave
    st.markdown(f'<div class="stat-box"><div class="stat-number">{will_attend}</div>Will Attend</div>', unsafe_allow_html=True)

# ---------------- CALCULATION ----------------
future_attended = attended + will_attend
future_total = total + remaining

attendance_percent = (future_attended / future_total) * 100

# ---------------- BUTTON ----------------
if st.button("Analyze Risk"):

    with st.spinner("Analyzing future attendance..."):
        time.sleep(1.2)

    st.markdown('<div class="card result warning">', unsafe_allow_html=True)

    st.markdown(f"<h1 style='color:#38bdf8'>{attendance_percent:.2f}%</h1>", unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width:{attendance_percent}%"></div>
    </div>
    """, unsafe_allow_html=True)

    if attendance_percent < 75:
        st.write("⚠️ You are below the safe threshold (75%). Improve attendance.")
    else:
        st.write("✅ You are safe. Keep it consistent.")

    st.markdown('</div>', unsafe_allow_html=True)
