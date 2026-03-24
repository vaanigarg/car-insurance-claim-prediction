import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

st.title("Car Insurance Claim Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df.head())

    df = df.select_dtypes(include=['int64', 'float64'])

    if "is_claim" not in df.columns:
        st.error("CSV must contain 'is_claim' column")
    else:
        X = df.drop("is_claim", axis=1)
        y = df["is_claim"]

        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            eval_metric='logloss'
        )
    
        model.fit(X,y)

        preds = model.predict(X)

        df["Prediction"] = preds

        st.write("Predictions", df)