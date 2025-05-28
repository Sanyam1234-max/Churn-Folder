import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load the best model
model = joblib.load('churn_model.pkl')

# Title
st.title("📉 Customer Churn Prediction")
st.markdown("Upload a customer dataset to predict the likelihood of churn.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Churn prediction function
def preprocess_and_predict(df):
    # Drop non-useful columns
    df = df.drop(['customerID'], axis=1, errors='ignore')

    # Convert TotalCharges if needed
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.fillna(0, inplace=True)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Align with training features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    # Make predictions
    preds = model.predict(df_encoded)
    probs = model.predict_proba(df_encoded)[:, 1]
    return preds, probs

# Handle file upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("🔍 Predict Churn"):
        try:
            preds, probs = preprocess_and_predict(df)
            df['Churn Prediction'] = preds
            df['Churn Probability'] = np.round(probs, 3)

            st.subheader("📊 Prediction Results")
            st.dataframe(df[['Churn Prediction', 'Churn Probability']])

            # Summary metrics
            churn_rate = np.mean(preds)
            st.success(f"🔻 Estimated churn rate: **{churn_rate:.2%}**")

            # Show histogram
            st.subheader("📈 Churn Probability Distribution")
            st.bar_chart(pd.Series(probs, name="Churn Probability"))

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and a Random Forest model.")
