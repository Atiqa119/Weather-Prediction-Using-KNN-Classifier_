import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("weather_data.csv")


X = df.drop("Weather Category", axis=1)
y = df["Weather Category"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Save the scaler and model for reuse
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

st.title("🌦 Weather Category Predictor (KNN)")

# User inputs
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

# Predict button
if st.button("Predict Weather Category"):
    # Load saved model and scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)

    # Prepare new data for prediction
    new_data = [[temperature, humidity, wind_speed, visibility]]
    new_data_scaled = scaler.transform(new_data)

    # Predict
    prediction = knn.predict(new_data_scaled)[0]

    # Show result
    st.success(f"🌤 Predicted Weather Category: **{prediction}**")
