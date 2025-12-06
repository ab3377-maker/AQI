import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load trained pipeline (preprocess + RandomForest)
model = joblib.load("aqi_rf_time_model.pkl")

st.title("Air Quality Index (AQI) Prediction")

st.markdown("Provide station and location details to predict the AQI value.")

# ---- User inputs: match training features ----
air_pollutant = st.selectbox(
    "Air Pollutant",
    ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"]
)

year = st.number_input("Year", min_value=1970, max_value=2050, value=2025)

data_coverage = st.number_input("Data Coverage (%)", min_value=0.0, max_value=100.0, value=95.0)

latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=48.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=2.0)
altitude = st.number_input("Altitude (m)", value=100.0)

air_quality_station_type = st.selectbox(
    "Station Type",
    ["Traffic", "Background", "Industrial", "Unknown"]
)

air_quality_station_area = st.selectbox(
    "Station Area",
    ["Urban", "Suburban", "Rural", "Unknown"]
)

country = st.text_input("Country", "France")

# Raw concentration used to compute AQI during training
air_pollution_level = st.number_input(
    "Air Pollution Level (same units as training)",
    value=30.0
)

# If you trained with a 'cluster' feature, you can default to 0 or let user choose
cluster = st.number_input("Cluster label (optional)", min_value=0, value=0)

# ---- Prediction ----
if st.button("Predict AQI"):
    # Build single-row DataFrame with exact feature names used in training
    input_df = pd.DataFrame([{
        "Air Pollution Level": air_pollution_level,
        "Year": year,
        "Data Coverage": data_coverage,
        "Altitude": altitude,
        "Latitude": latitude,
        "Longitude": longitude,
        "Air Pollutant": air_pollutant,
        "Air Quality Station Type": air_quality_station_type,
        "Air Quality Station Area": air_quality_station_area,
        "Country": country,
        "cluster": cluster,
    }])

    pred_aqi = model.predict(input_df)[0]

    st.subheader(f"Predicted AQI value: {pred_aqi:.2f}")

    def aqi_band(aqi):
        if 0 <= aqi <= 50:
            return "Good"
        elif 51 <= aqi <= 100:
            return "Moderate"
        elif 101 <= aqi <= 150:
            return "Unhealthy for Sensitive"
        elif 151 <= aqi <= 200:
            return "Unhealthy"
        elif 201 <= aqi <= 300:
            return "Very Unhealthy"
        elif 301 <= aqi <= 500:
            return "Hazardous"
        else:
            return "Out of Range"

    st.write("AQI band:", aqi_band(pred_aqi))
