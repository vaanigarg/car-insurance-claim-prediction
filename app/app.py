import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/xgb_model_v2.pkl")
columns = joblib.load("models/columns.pkl")

st.title("Car Insurance Claim Prediction")

st.write("Upload input to predict claim probability")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df = input_df.reindex(columns=columns, fill_value=0)
    st.write("### Uploaded Data Preview")
    st.dataframe(input_df.head())

    try:
        predictions = model.predict(input_df)

        input_df["Prediction"] = predictions

        st.write("### Predictions")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error: {e}")