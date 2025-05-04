import streamlit as st
import pickle
import numpy as np

# Load the trained SVC model
with open('svc_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("🌸 Iris Flower Species Prediction")
st.write("Enter the flower features below to predict its species (Setosa = 0, Versicolor = 1, Virginica = 2).")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    
    # Map class to species name (optional)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    species_name = species_map.get(prediction, "Unknown")
    
    st.success(f"Predicted Species: {species_name} (Class {prediction})")