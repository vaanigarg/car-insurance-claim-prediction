import streamlit as st
import joblib
import os
import pandas as pd

st.set_page_config(page_title="Car Insurance Prediction", layout="wide")

st.title("Car Insurance Claim Prediction")

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "xgb_model.pkl")
    model = joblib.load(model_path)
    return model

@st.cache_resource
def load_columns():
    col_path = os.path.join("models", "columns.pkl")
    return joblib.load(col_path)

model = load_model()
columns = load_columns()

uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(input_df.head())

    try:

        input_df = input_df.reindex(columns=columns, fill_value=0)

        predictions = model.predict(input_df)

        input_df["Prediction"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")