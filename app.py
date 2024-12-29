import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Diabetes Prediction App")

# Input fields for user data
st.sidebar.header("Enter Patient Data")
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=800, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# Create a numpy array from inputs
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# Predict button
if st.sidebar.button("Predict"):
    prediction = model.predict(user_input)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.write(f"The patient is **{result}**.")
