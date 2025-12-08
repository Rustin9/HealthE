import numpy as np
import pandas as pd
import joblib
regression_model = joblib.load("models/LightGBM_recovery_time.joblib")
scaler = joblib.load("models/recovery_scaler_realistic.joblib")

example_input_recovery = {
        "age": 28,
        "gender": 1,
        "bmi": 24.5,
        'condition_type': 3,
        'severity_score': 5,
        'rest_hours_per_day': 7,
        'medication_adherence': 0.5,
        'hospital_visits': 2,
        'smoking_status': 0
    }
df = pd.DataFrame([example_input_recovery])
cols_to_scale = [
'age',"gender", 'bmi', 'condition_type','severity_score',
'rest_hours_per_day', 'medication_adherence', 'hospital_visits','smoking_status'
]
# Scale only selected columns
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
#print(df_scaled)
prediction = regression_model.predict(df_scaled)[0]
print(str(prediction.round(2)) +" days")
