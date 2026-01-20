import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Visa Processing Time Estimator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI-Enabled Visa Processing Time Estimator")
st.write("Predict visa processing time using trained ML models")

# Check model file
if not os.path.exists("processing_time_model.pkl"):
    st.error("❌ Model file not found. Please upload processing_time_model.pkl")
    st.stop()

model = joblib.load("processing_time_model.pkl")

st.sidebar.header("Applicant Details")
employees = st.sidebar.number_input("Number of Employees", min_value=1, value=50)
company_age = st.sidebar.number_input("Company Age (Years)", min_value=1, value=5)
annual_wage = st.sidebar.number_input("Annual Wage (USD)", min_value=1000, value=60000)

if st.button("🔮 Predict Processing Time"):
    input_data = np.array([[employees, company_age, annual_wage]])
    prediction = int(model.predict(input_data)[0])
    st.success(f"Estimated Processing Time: {prediction} days")

st.markdown("---")
st.markdown("© 2026 | AI-Enabled Visa Prediction System")
