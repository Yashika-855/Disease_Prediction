import streamlit as st
import numpy as np
import pickle
import os

# -----------------------------
# Load models safely
# -----------------------------
try:
    reg = pickle.load(open(os.path.join("reg_model.pkl"), "rb"))
    clf = pickle.load(open(os.path.join("clf_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join("scaler.pkl"), "rb"))
except Exception as e:
    st.error("❌ Error loading model files. Make sure .pkl files are uploaded correctly.")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Medical Prediction App")

st.title("🏥 Medical Prediction App")
st.header("Enter Patient Details")

# Inputs
age = st.slider("Age", 0, 100, 25)
bmi = st.slider("BMI", 10.0, 50.0, 22.0)
children = st.slider("Children", 0, 5, 0)

gender = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
insurance = st.selectbox("Insurance", ["Basic", "Premium"])
city = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])

# Health inputs
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

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict 💰"):

    input_data = np.array([[ 
        age, gender, bmi, children, smoker,
        activity, insurance, city,
        diabetes, hypertension, heart, asthma
    ]])

    try:
        input_scaled = scaler.transform(input_data)

        cost = reg.predict(input_scaled)[0]
        disease = clf.predict(input_scaled)[0]

        st.success(f"💰 Estimated Cost: ₹ {round(cost, 2)}")

        if disease == 1:
            st.error("⚠️ Risk of Disease Detected")
        else:
            st.success("✅ No Disease Risk")

    except Exception as e:
        st.error("❌ Prediction failed. Check model compatibility.")