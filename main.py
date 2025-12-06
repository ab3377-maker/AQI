from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import joblib
import pandas as pd

# ---------- 1. Download model from Google Drive at startup ----------

# Your shared link:
# https://drive.google.com/file/d/1Zbk7do-a4ussAX1qsad_k83v09jbve0u/view?usp=sharing
# Direct download URL pattern for Google Drive:
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Zbk7do-a4ussAX1qsad_k83v09jbve0u"
MODEL_PATH = "aqi_rf_time_model.pkl"


def load_model():
    # Download once if not present
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded to:", MODEL_PATH)
    else:
        print("Model file already exists:", MODEL_PATH)

    print("Loading model...")
    model_obj = joblib.load(MODEL_PATH)
    print("Model loaded.")
    return model_obj


# ---------- 2. FastAPI app and schemas ----------

class AQIRequest(BaseModel):
    Air_Pollutant: str
    Year: int
    Data_Coverage: float
    Altitude: float
    Latitude: float
    Longitude: float
    Air_Quality_Station_Type: str
    Air_Quality_Station_Area: str
    Country: str
    Air_Pollution_Level: float
    cluster: int = 0


def aqi_band_from_value(aqi: float) -> str:
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


app = FastAPI()

# Allow all origins (you can restrict to your frontend origin later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or ["https://your-frontend-domain"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = load_model()


# ---------- 3. Prediction endpoint ----------

@app.post("/predict")
def predict(req: AQIRequest):
    # Build DataFrame with exact column names used in training
    df = pd.DataFrame([{
        "Air Pollution Level": req.Air_Pollution_Level,
        "Year": req.Year,
        "Data Coverage": req.Data_Coverage,
        "Altitude": req.Altitude,
        "Latitude": req.Latitude,
        "Longitude": req.Longitude,
        "Air Pollutant": req.Air_Pollutant,
        "Air Quality Station Type": req.Air_Quality_Station_Type,
        "Air Quality Station Area": req.Air_Quality_Station_Area,
        "Country": req.Country,
        "cluster": req.cluster,
    }])

    # Predict AQI_value using the loaded pipeline
    pred = model.predict(df)[0]
    band = aqi_band_from_value(pred)

    return {
        "AQI_value": float(pred),
        "AQI_band": band,
    }
