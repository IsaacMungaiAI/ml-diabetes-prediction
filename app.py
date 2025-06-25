import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("diabetes_model2.pkl")
scaler=joblib.load("scaler.pkl")

# App title
st.title("🩺 Diabetes Prediction App")

# User input form
st.subheader("Enter patient data:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Predict button
if st.button("Predict"):
    # Make prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ The patient is likely to have diabetes.")
    else:
        st.success("✅ The patient is not likely to have diabetes.")
