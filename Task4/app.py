import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained Random Forest model and scaler
rf_model = joblib.load('Sales1_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("Sales Prediction App")
st.write("Enter the marketing budget data to predict the sales.")

# Collect marketing budget details from the user
st.subheader("Enter Marketing Budget Details")
tv_budget = st.number_input("TV Advertising Budget (in thousands of dollars):", min_value=0.0)
radio_budget = st.number_input("Radio Advertising Budget (in thousands of dollars):", min_value=0.0)
newspaper_budget = st.number_input("Newspaper Advertising Budget (in thousands of dollars):", min_value=0.0)


# Prepare input data for prediction
input_data = pd.DataFrame({
    'TV': [tv_budget],
    'Radio': [radio_budget],
    'Newspaper': [newspaper_budget]
})

# Scale the input data using the same scaler
input_data_scaled = scaler.transform(input_data)

# Make predictions
if st.button("Predict Sales"):
    prediction = rf_model.predict(input_data_scaled)
    st.write(f"Predicted Sales: {prediction[0]} units")
