import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========= Page Config =========
st.set_page_config(
    page_title="Healthcare ML Predictions",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========= Load Models =========
@st.cache_resource
def load_models():
    recovery_model = joblib.load("models/LightGBM_recovery_time.joblib")
    diet_model = joblib.load("models/lr_diet_model_realistic.joblib")
    recovery_scaler = joblib.load("models/recovery_scaler_realistic.joblib")
    diet_scaler = joblib.load("models/diet_scaler_realistic.joblib")
    return recovery_model, diet_model, recovery_scaler, diet_scaler

recovery_model, diet_model, recovery_scaler, diet_scaler = load_models()

# ========= Custom CSS =========
st.markdown("""
<style>
body {
    background-color: #F5F9FF;
}
.main-header {
    text-align: center;
    padding: 10px;
    color: #0A4D68;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #D0E4FF;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ========= Header =========
st.markdown("<h1 class='main-header'>🏥 Healthcare Prediction App</h1>", unsafe_allow_html=True)
#st.write("Use AI-powered prediction models to estimate **Recovery Rates** and recommend **Diet Plans**.")

# ========= Sidebar Navigation =========
sidebar_choice = st.sidebar.radio(
    "Choose Prediction Type",
    ["🔧 Predict Recovery Days", "🍎 Predict Diet Plan"]
)

# -------------------------------------------------------------
# 📌 MODEL 1 — RECOVERY RATE PREDICTION
# -------------------------------------------------------------
if sidebar_choice == "🔧 Predict Recovery Days":

    #st.markdown("<div class='card'><h2>🔧 Recovery Rate Prediction</h2></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown(
            """
            <div class="recovery-card">
                <h2>🔧 Recovery Days Prediction</h2>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <style>
        .recovery-card {
            background-color: #263238;
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 8px solid #4caf50;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            text-align: center;
            font-size: 1.5rem;
            color: #e0f7fa;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.slider("BMI", 12.0, 45.0, 26.5)
    condition_type = st.selectbox("Current Condition", ["Flu", "Infection", "Allergy", "Fever", "Cough", "Injury"])
    severity_score = st.slider("Severity Score", 1.0, 10.0, 5.0)
    rest_hours_per_day = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    medication_adherence = st.slider("Medical Adherence", 0.0, 1.0, 0.5)
    hospital_visits = st.slider("Hospital Visits", 0, 7, 0)
    smoking_status = st.selectbox("Smoking Status",["Non-Smoker","Occasional","Regular"])


    # Convert categorical → numeric
    condition_map = {"Allergy":0 ,"Cough": 1, "Fever": 2, "Flu": 3, "Infection": 4, "Injury": 5}
    gender_map = {"Male":1, "Female":0, "Other":2}
    smoking_status_map = {"Non-Smoker":0, "Occasional":1, "Regular":2}

    X = pd.DataFrame({
        "age": [age],
        "gender": [gender_map[gender]],
        "bmi": [bmi],
        "condition_type": [condition_map[condition_type]],
        "severity_score": [severity_score],
        "rest_hours_per_day": [rest_hours_per_day],
        "medication_adherence":[medication_adherence],
        "hospital_visits": [hospital_visits],
        "smoking_status": [smoking_status_map[smoking_status]]
    })

    # Scale only numeric columns used during training
    X_scaled = recovery_scaler.transform(X)

    if st.button("Predict Recovery Rate"):
        pred = recovery_model.predict(X_scaled)[0]
        st.success(f"🩺 **Predicted Recovery Days: {round(pred,2)} days**")

# -------------------------------------------------------------
# 📌 MODEL 2 — DIET PLAN PREDICTION
# -------------------------------------------------------------
if sidebar_choice == "🍎 Predict Diet Plan":

    #st.markdown("<div class='card'><h2>🍎 Diet Plan Recommendation</h2></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown(
            """
            <div class="recovery-card">
                <h2>🍎 Diet Plan Recommendation</h2>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <style>
        .recovery-card {
            background-color: #e0f2f1;
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 8px solid #4caf50;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            text-align: center;
            font-size: 1.5rem;
            color: #00695c;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    bmi = st.slider("BMI", 12.0, 45.0, 26.5)
    calories = st.slider("Daily Calorie Intake", 1000, 4500, 2500)
    protein = st.slider("Daily Protein Intake (g)", 20, 250, 90)
    carbs = st.slider("Daily Carbs Intake (g)", 50, 450, 180)
    fats = st.slider("Daily Fat Intake (g)", 10, 150, 60)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    daily_steps = st.slider("Daily Steps", 1000, 18000, 1500)
    water_intake_liters = st.slider("Water Intake (Litres)", 1, 5, 3)
    conditions = st.selectbox("Current Condition", ["Flu", "Infection", "Allergy", "Fever", "Cough", "Injury"])
    conditions_map = {"Allergy": 0, "Cough": 1, "Fever": 2, "Flu": 3, "Infection": 4, "Injury": 5}
    gender_map = {"Female": 0, "Male": 1, "Other":2}

    X = pd.DataFrame({
        "age": [age],
        "gender": [gender_map[gender]],
        "conditions": [conditions_map[conditions]],
        "bmi": [bmi],
        "daily_calories": [calories],
        "protein_intake": [protein],
        "carb_intake": [carbs],
        "fat_intake": [fats],
        "sleep_hours": [sleep_hours],
        "daily_steps": [daily_steps],
        "water_intake_liters": [water_intake_liters]
    })

    X_scaled = diet_scaler.transform(X)

    if st.button("Recommend Diet Plan"):
        pred_class = diet_model.predict(X_scaled)[0]
        pred_label = {0:"Balanced Diet", 1:"High-Protein Diet", 2:"Low-Carb Diet"}
        st.success(f"🍏 **Recommended Diet Plan: {pred_label[pred_class]}**")

        st.info("""
### Plan Explanation
This recommendation is based on your calorie intake, BMI, health condition,
and macronutrient balance. A detailed report can be generated in the next update.
""")

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("##### Built with ❤️ by DC Students")
