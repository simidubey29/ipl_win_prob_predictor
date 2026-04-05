import streamlit as st

st.set_page_config(page_title="IPL Win Predictor", layout="centered")
st.title("IPL Predictor")   
import pickle
import numpy as np


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# =========================
# PAGE CONFIG
# =========================


# =========================
# TITLE
# =========================
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏏 IPL Win Probability Predictor</h1>",
            unsafe_allow_html=True)
st.markdown("---")

# =========================
# TEAM SELECTION
# =========================
teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Punjab Kings"
]

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)

with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", teams)

# =========================
# INNINGS SELECTION
# =========================
st.markdown("---")
inning = st.selectbox("🧢 Select Innings", ["1st Innings", "2nd Innings"])

if batting_team == bowling_team:
    st.warning("⚠️ Batting and Bowling team cannot be same")

st.markdown("---")

# =========================
# MATCH INPUTS
# =========================
st.markdown("### 📊 Match Situation")

col1, col2 = st.columns(2)

with col1:
    score = st.number_input("Current Score", min_value=0, value=100)
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3)

with col2:
    overs = st.number_input("Overs Completed", min_value=0.1, max_value=20.0, value=10.0)

    # Target only for 2nd innings
    if inning == "2nd Innings":
        target = st.number_input("Target Score", min_value=1, value=180)
    else:
        target = 200  # dummy target for 1st innings

# =========================
# CALCULATIONS
# =========================
balls_left = int(120 - (overs * 6))
wickets_left = 10 - wickets

overs_completed = overs if overs > 0 else 1

crr = score / overs_completed

# For 2nd innings
if inning == "2nd Innings":
    runs_left = target - score
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
else:
    runs_left = 0
    rrr = 0

# =========================
# PREDICTION BUTTON
# =========================
st.markdown("---")

if st.button("🔮 Predict Win Probability"):

    if batting_team == bowling_team:
        st.error("❌ Please select different teams")

    else:
        # For 1st innings → prediction not meaningful
        if inning == "1st Innings":
            st.info("ℹ️ Win prediction is more accurate in 2nd innings (chasing scenario).")

        input_data = np.array([[score, balls_left, wickets_left, crr, rrr]])

        prob = model.predict_proba(input_data)[0]

        win_prob = round(prob[1] * 100, 2)
        lose_prob = round(prob[0] * 100, 2)

        # =========================
        # RESULT UI
        # =========================
        st.markdown("## 📈 Match Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"🏏 {batting_team} Win Probability")
            st.metric(label="Win %", value=f"{win_prob}%")

        with col2:
            st.error(f"🎯 {bowling_team} Win Probability")
            st.metric(label="Win %", value=f"{lose_prob}%")

        st.progress(int(win_prob))

        # =========================
        # EXTRA MATCH INFO
        # =========================
        st.markdown("---")
        st.markdown("### 📌 Match Stats")

        st.write(f"Balls Left: {balls_left}")
        st.write(f"Wickets Left: {wickets_left}")
        st.write(f"Current Run Rate (CRR): {round(crr, 2)}")

        if inning == "2nd Innings":
            st.write(f"Runs Left: {runs_left}")
            st.write(f"Required Run Rate (RRR): {round(rrr, 2)}")

st.markdown("---")
st.markdown(
    "<h4 style='text-align: center;'>✨ Built with ❤️ by <b>Simi Dubey</b></h4>",
    unsafe_allow_html=True
)