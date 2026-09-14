import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Air Quality MLOps API")

model = joblib.load("models/linear_regression_pipeline.pkl")


class AirQualityInput(BaseModel):
    pm10: float
    carbon_monoxide: float
    nitrogen_dioxide: float
    sulphur_dioxide: float
    ozone: float
    hour: int
    day_of_week: int
    month: int
    pm2_5_lag_1: float
    pm2_5_lag_3: float
    pm2_5_lag_24: float


@app.get("/")
def root():
    return {"message": "Air Quality API is running"}


@app.post("/predict")
def predict(data: AirQualityInput):
    input_data = pd.DataFrame([{
    "pm10": data.pm10,
    "carbon_monoxide": data.carbon_monoxide,
    "nitrogen_dioxide": data.nitrogen_dioxide,
    "sulphur_dioxide": data.sulphur_dioxide,
    "ozone": data.ozone,
    "hour": data.hour,
    "day_of_week": data.day_of_week,
    "month": data.month,
    "pm2_5_lag_1": data.pm2_5_lag_1,
    "pm2_5_lag_3": data.pm2_5_lag_3,
    "pm2_5_lag_24": data.pm2_5_lag_24,
}])

    prediction = model.predict(input_data)[0]

    return {"predicted_pm2_5": float(prediction)}