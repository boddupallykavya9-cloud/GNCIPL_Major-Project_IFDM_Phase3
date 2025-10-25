import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import date, datetime

# 1) Load the trained model and scaler
# Ensure final_model.pkl exists in the same directory
with open('final_model.pkl', 'rb') as f:
    model, scaler = pickle.load(f)

# 2) App header
st.title("Insurance Fraud Detection")

# 3) Feature inputs
# Customize these to match exactly what your model expects (names, types, order)
# Example subset (update to full feature set from training)
# If you trained with claim_delay, include a date-based input; otherwise compute it.

# Core numeric features
age = st.number_input("Age", value=30, min_value=0, max_value=120, step=1)
children = st.number_input("Children", value=0, min_value=0, step=1)
bmi = st.number_input("BMI", value=25.0, min_value=0.0, step=0.1)
bill_amount = st.number_input("Bill Amount", value=1000.0, min_value=0.0, step=0.01)
claimed_amount = st.number_input("Claimed Amount", value=800.0, min_value=0.0, step=0.01)
amount_paid = st.number_input("Amount Paid", value=700.0, min_value=0.0, step=0.01)
duration = st.number_input("Duration (days)", value=12, min_value=0, step=1)
year_billing = st.number_input("Year Billing", value=2024, min_value=1900, max_value=2100, step=1)

# Categorical inputs (you might have encoded them as integers in training)
# If you used LabelEncoder in training, you must reproduce the same mapping.
sex = st.selectbox("Sex", ["Male","Female"])
region = st.selectbox("Region", ["southeast","southwest","northeast","northwest"])
smoker = st.selectbox("Smoker", ["Yes","No"])
# Optional: include region_code or other encodings if used in training
# We'll map common ones; replace with exact mapping used in your training if different.

sex_map = {"Male": 1, "Female": 0}
region_map = {"southeast":0, "southwest":1, "northeast":2, "northwest":3}
smoker_map = {"Yes": 1, "No": 0}

# Optional date inputs to compute claim_delay
apply_date = st.date_input("Insurance Apply Date", value=date(2024,1,1))
claimed_date = st.date_input("Insurance Claimed Date", value=date(2024,1,2))
claim_delay = (pd.Timestamp(claimed_date) - pd.Timestamp(apply_date)).days

# Display derived value for user awareness
st.write("Claim Delay (days):", claim_delay)

# Build input vector in the exact order your model expects
# Update the order to exactly match training feature order
# Example assumed order (adjust as needed):
input_features = [
    age,
    children,
    sex_map[sex],
    region_map[region],
    bmi,
    smoker_map[smoker],
    bill_amount,
    claimed_amount,
    amount_paid,
    duration,
    year_billing,
    claim_delay
]

# 4) Prediction
user_data = np.array(input_features).reshape(1, -1)
if st.button("Predict"):
    # Scale using the same scaler used during training
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(user_data_scaled)[:, 1][0]
    result = "Fraudulent Claim" if int(prediction[0]) == 1 else "Legitimate Claim"
    st.write("Prediction:", result)
    if prob is not None:
        st.write(f"Fraud probability: {prob:.3f}")

# 5) Optional: visualize or export results

