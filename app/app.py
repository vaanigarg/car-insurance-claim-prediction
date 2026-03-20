import streamlit as st
import joblib
import pandas as pd

model = joblib.load("../models/final_model.pkl")

st.title("Car Insurance Claim Prediction")

age_of_car = st.number_input("Age of Car")
population_density = st.number_input("Population Density")

input_df = pd.DataFrame({
    "age_of_car": [age_of_car],
    "population_density": [population_density]
})

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write("Prediction:", prediction[0])