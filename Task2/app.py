import streamlit as st
import numpy as np
import pickle

# Load the trained XGBoost model
model = pickle.load(open("best_xgboost_model.pkl", "rb"))

st.title("🎬 Movie Rating Predictor (XGBoost)")
st.markdown("Enter the movie details below to predict the rating.")

# 🛠️ Replace these with your actual feature inputs
movie_name = st.text_input("Enter Movie Name")
release_year = st.number_input("Enter Release Year", min_value=1900, max_value=2025, value=2020)
duration = st.number_input("Enter Duration (in minutes)", value=120)
genre = st.text_input("Enter Genre(s) (comma separated, e.g., Drama, Action)")
director = st.text_input("Enter Director")
actor1 = st.text_input("Enter Actor 1")
actor2 = st.text_input("Enter Actor 2")
actor3 = st.text_input("Enter Actor 3")

# You can add more features depending on the columns your model expects.

# Collect input and make prediction
if st.button("🚀 Predict Rating"):
    try:
        # Preprocess and convert inputs as necessary (e.g., encoding categorical data)
        input_data = np.array([[release_year, duration]])  # Shape must match model input
        # Add other features here if needed (e.g., genre encoding, actor encoding, etc.)
        
        prediction = model.predict(input_data)[0]  # Make prediction
        st.success(f"⭐ Predicted Rating: {prediction:.2f}")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")
