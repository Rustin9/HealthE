import os

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ========= Page Config =========
st.set_page_config(
    page_title="HealthE",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========= Backend URL (for logging & history) =========
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ========= Load Models =========
@st.cache_resource
def load_models():
    recovery_model = joblib.load("models/LightGBM_recovery_time.joblib")
    diet_model = joblib.load("models/new_lr_model_final.joblib")
    recovery_scaler = joblib.load("models/recovery_scaler_realistic.joblib")
    diet_scaler = joblib.load("models/new_diet_scaler_final.joblib")
    return recovery_model, diet_model, recovery_scaler, diet_scaler


recovery_model, diet_model, recovery_scaler, diet_scaler = load_models()


def _to_builtin(val):
    """Convert NumPy scalars to native Python types so JSON serialization works."""
    if isinstance(val, (np.generic,)):
        return val.item()
    return val


def send_log_to_backend(prediction_type, inputs, output):
    """
    Fire-and-forget logging of a prediction to the backend.
    Does NOT break the UI if backend is down.
    """
    clean_inputs = {k: _to_builtin(v) for k, v in inputs.items()}
    clean_output = {k: _to_builtin(v) for k, v in output.items()}

    payload = {
        "prediction_type": prediction_type,
        "inputs": clean_inputs,
        "output": clean_output,
    }

    try:
        requests.post(
            f"{BACKEND_URL}/log_prediction",
            json=payload,
            timeout=2,
        )
    except Exception:
        # ignore logging errors to keep UX smooth
        pass


# ========= Custom CSS =========
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: radial-gradient(circle at top left, #f5fbff 0, #f3f4f8 40%, #e9ecf2 100%);
    }

    /* Main content container tweaks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] label {
        color: #e5e7eb;
    }

    /* Card-like container for forms */
    .prediction-card {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 1rem;
        padding: 1.5rem 1.75rem;
        border: 1px solid rgba(148, 163, 184, 0.45);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(4px);
    }

    /* Primary buttons */
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 25px rgba(22, 163, 74, 0.35);
        background: linear-gradient(135deg, #16a34a, #15803d);
    }

    /* Input labels a bit bolder */
    label[data-testid="stWidgetLabel"] > div {
        font-weight: 600;
    }

    /* Footer text */
    .he-footer {
        color: #6b7280;
        text-align: center;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========= Header =========
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 1.5rem;">
        <h1 style="margin-bottom:0.2rem;">🩺 HealthE</h1>
        <p style="color:#4b5563; font-size:0.98rem;">
            AI-powered wellness & recovery assistant – predict recovery time and get a tailored diet suggestion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========= Sidebar Navigation =========
sidebar_choice = st.sidebar.radio(
    "Choose Prediction Type",
    ["⏱️ Predict Recovery Days", "🥗 Predict Diet Plan"],
)

# ========= Sidebar: recent history from backend DB =========
with st.sidebar.expander("🧾 Recent Predictions (from DB)"):
    try:
        resp = requests.get(
            f"{BACKEND_URL}/history", params={"limit": 10}, timeout=2
        )
        if resp.status_code == 200:
            records = resp.json()
            if not records:
                st.write("No predictions logged yet.")
            else:
                rows = []
                for r in records:
                    rows.append(
                        {
                            "type": r["prediction_type"],
                            "created_at": r["created_at"],
                            "output": r["output"],
                        }
                    )
                df_hist = pd.DataFrame(rows)
                st.dataframe(df_hist, use_container_width=True)
        else:
            st.write("Could not load history.")
    except Exception:
        st.write("Backend not reachable.")

# -------------------------------------------------------------
# MODEL 1 — RECOVERY RATE PREDICTION
# -------------------------------------------------------------
if sidebar_choice == "⏱️ Predict Recovery Days":
    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        st.markdown("## 🕒 Recovery Days Prediction")

        # Layout the inputs in two columns for a cleaner UI
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 1, 100, 25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            bmi = st.slider("BMI", 12.0, 45.0, 26.5)
            condition_type = st.selectbox(
                "Current Condition",
                ["Flu", "Infection", "Allergy", "Fever", "Cough", "Injury"],
            )
            severity_score = st.slider("Severity Score", 1.0, 10.0, 5.0)

        with col2:
            rest_hours_per_day = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
            medication_adherence = st.slider(
                "Medical Adherence", 0.0, 1.0, 0.5
            )
            hospital_visits = st.slider("Hospital Visits", 0, 7, 0)
            smoking_status = st.selectbox(
                "Smoking Status", ["Non-Smoker", "Occasional", "Regular"]
            )

        # Convert categorical → numeric
        condition_map = {
            "Allergy": 0,
            "Cough": 1,
            "Fever": 2,
            "Flu": 3,
            "Infection": 4,
            "Injury": 5,
        }
        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        smoking_status_map = {
            "Non-Smoker": 0,
            "Occasional": 1,
            "Regular": 2,
        }

        X = pd.DataFrame(
            {
                "age": [age],
                "gender": [gender_map[gender]],
                "bmi": [bmi],
                "condition_type": [condition_map[condition_type]],
                "severity_score": [severity_score],
                "rest_hours_per_day": [rest_hours_per_day],
                "medication_adherence": [medication_adherence],
                "hospital_visits": [hospital_visits],
                "smoking_status": [smoking_status_map[smoking_status]],
            }
        )

        # Scale only numeric columns used during training
        X_scaled = recovery_scaler.transform(X)

        if st.button("Predict Recovery Days"):
            pred = recovery_model.predict(X_scaled)[0]
            pred_rounded = round(float(pred), 2)

            st.success(f"**Predicted Recovery Days: {pred_rounded} days**")

            # Log to backend (non-blocking)
            send_log_to_backend(
                "recovery_days",
                inputs=X.iloc[0].to_dict(),
                output={"recovery_days": pred_rounded},
            )

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODEL 2 — DIET PLAN PREDICTION
# -------------------------------------------------------------
if sidebar_choice == "🥗 Predict Diet Plan":
    with st.container():
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        st.markdown("## 🥗 Diet Plan Recommendation")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 1, 100, 25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            conditions = st.selectbox(
                "Current Condition",
                ["Flu", "Infection", "Allergy", "Fever", "Cough", "Injury", "No"],
            )
            bmi = st.slider("BMI", 12.0, 45.0, 26.5)
            calories = st.slider("Daily Calorie Intake", 1000, 4500, 2500)
            protein = st.slider("Daily Protein Intake (g)", 20, 250, 90)

        with col2:
            carbs = st.slider("Daily Carbs Intake (g)", 50, 450, 180)
            fats = st.slider("Daily Fat Intake (g)", 10, 150, 60)
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
            daily_steps = st.slider("Daily Steps", 1000, 18000, 1500)
            water_intake_liters = st.slider("Water Intake (Litres)", 1, 5, 3)

        conditions_map = {
            "Allergy": 0,
            "Cough": 1,
            "Fever": 2,
            "Flu": 3,
            "Infection": 4,
            "Injury": 5,
            "No": 6,
        }
        gender_map = {"Male": 1, "Female": 0}

        X = pd.DataFrame(
            {
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
                "water_intake_liters": [water_intake_liters],
            }
        )

        cols_to_scale = [
            "age",
            "bmi",
            "daily_calories",
            "protein_intake",
            "carb_intake",
            "fat_intake",
            "sleep_hours",
            "daily_steps",
            "water_intake_liters",
        ]

        X_scaled = X.copy()
        X_scaled[cols_to_scale] = diet_scaler.transform(X[cols_to_scale])

        if st.button("Recommend Diet Plan"):
            pred_class = int(diet_model.predict(X_scaled)[0])
            pred_label = {
                0: "Balanced Diet",
                1: "High-Protein Diet",
                2: "Keto Diet",
                3: "Low-Carb Diet",
                4: "Low-Fat Diet",
                5: "Vegan Diet",
            }
            label = pred_label.get(pred_class, "Unknown")

            st.success(f"**Recommended Diet Plan: {label}**")

            # Log to backend (non-blocking)
            send_log_to_backend(
                "diet_plan",
                inputs=X.iloc[0].to_dict(),
                output={
                    "diet_plan_class": pred_class,
                    "diet_plan_label": label,
                },
            )

            st.info(
                """
                ### Plan Explanation
                This recommendation is based on your calorie intake,
                BMI, health condition, and macronutrient balance.
                A more detailed report can be added in a future update.
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p class="he-footer">Built with ❤️ by DC Students • UI tweaks contributed by the community</p>',
    unsafe_allow_html=True,
)
