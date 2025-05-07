import streamlit as st
import numpy as np
import joblib

# Load the logistic regression model
model = joblib.load("logistic_model.pkl")

# Optional: Load the scaler if you used one during training
# scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to predict if it's fraudulent:")

# Input form for 30 features (you can customize labels as needed)
input_features = []
for i in range(30):
    feature = st.number_input(f"Feature {i+1}", value=0.0, step=0.01, format="%.6f")
    input_features.append(feature)

# When user clicks the Predict button
if st.button("Predict"):
    # Prepare input data
    features = np.array(input_features).reshape(1, -1)

    # Uncomment below if you used a scaler
    # features = scaler.transform(features)

    prediction = model.predict(features)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
