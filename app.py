import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Medical App", page_icon="🌙", layout="centered")

# -----------------------------
# Dark Theme CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
    text-align: center;
}
.stButton>button {
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(to right, #0891b2, #2563eb);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
def load_model(file):
    if not os.path.exists(file):
        st.error(f"❌ Missing file: {file}")
        st.stop()
    return pickle.load(open(file, "rb"))

reg = load_model("reg_model.pkl")
clf = load_model("clf_model.pkl")
scaler = load_model("scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🌙 Medical Prediction Dashboard")
st.write("Predict **Medical Cost** and **Disease Risk**")

st.divider()

# -----------------------------
# Input UI
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 25)
    bmi = st.slider("BMI", 10.0, 50.0, 22.0)
    children = st.slider("Children", 0, 5, 0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])
    activity = st.selectbox("Activity", ["Low", "Medium", "High"])

col3, col4 = st.columns(2)

with col3:
    insurance = st.selectbox("Insurance", ["Basic", "Premium"])
    city = st.selectbox("City", ["Urban", "Semi-Urban", "Rural"])

with col4:
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart = st.selectbox("Heart Disease", [0, 1])
    asthma = st.selectbox("Asthma", [0, 1])

# -----------------------------
# Encoding
# -----------------------------
gender = 1 if gender == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

activity_map = {"Low": 0, "Medium": 1, "High": 2}
insurance_map = {"Basic": 0, "Premium": 1}
city_map = {"Urban": 0, "Semi-Urban": 1, "Rural": 2}

activity = activity_map[activity]
insurance = insurance_map[insurance]
city = city_map[city]

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Now"):

    input_data = np.array([[ 
        age, gender, bmi, children, smoker,
        activity, insurance, city,
        diabetes, hypertension, heart, asthma
    ]])

    input_scaled = scaler.transform(input_data)

    cost = reg.predict(input_scaled)[0]
    disease = clf.predict(input_scaled)[0]

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("📊 Prediction Results")

    col5, col6 = st.columns(2)

    with col5:
        st.metric("💰 Estimated Cost", f"₹ {round(cost,2)}")

    with col6:
        if disease == 1:
            st.error("⚠️ High Disease Risk")
        else:
            st.success("✅ Low Disease Risk")

    # -----------------------------
    # Graph (Feature Impact Style)
    # -----------------------------
    st.subheader("📈 Input Overview")

    labels = ["Age", "BMI", "Children", "Cost"]
    values = [age, bmi, children, cost]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    st.pyplot(fig)

    st.balloons()