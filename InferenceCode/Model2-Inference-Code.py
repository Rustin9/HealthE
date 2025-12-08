import numpy as np
import pandas as pd
import joblib

classification_model = joblib.load("new_lr_model_final.joblib")
scaler = joblib.load("new_diet_scaler_final.joblib")

# Columns used for scaling
cols_to_scale = [
    "age", "bmi", "daily_calories", "protein_intake",
    "carb_intake", "fat_intake", "sleep_hours",
    "daily_steps", "water_intake_liters"
]

def predict_diet_plan(input_dict):
    """
    Predict diet plan using Logistic Regression classification model.
    """

    df = pd.DataFrame([input_dict])

    # Scale only selected columns
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Predict diet class
    pred_class = classification_model.predict(df_scaled)[0]
    #pred_proba = classification_model.predict_proba(df_scaled)[0]

    return {
        "predicted_class": pred_class
    }


example_input_diet = {
        "age": 28,
        "gender": 0,
        "conditions": 1,
        "bmi": 24.49,
        "daily_calories": 2500,
        "protein_intake": 119,
        "carb_intake": 220,
        "fat_intake": 70,
        "sleep_hours": 7,
        "daily_steps": 9009,
        "water_intake_liters": 3
    }


diet_class = predict_diet_plan(example_input_diet)
pred_label = {0: 'Balanced Diet',1: 'High-Protein Diet',2: 'Keto Diet',3: 'Low-Carb Diet',4: 'Low-Fat Diet',5: 'Vegan Diet'}
label = pred_label.get(diet_class["predicted_class"],"Unk")
print("Diet Plan Prediction: "+ label)