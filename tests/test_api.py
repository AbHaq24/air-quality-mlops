from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Air Quality API is running"}


def test_predict():
    payload = {
        "pm10": 40.0,
        "carbon_monoxide": 250.0,
        "nitrogen_dioxide": 20.0,
        "sulphur_dioxide": 5.0,
        "ozone": 80.0,
        "hour": 12,
        "day_of_week": 2,
        "month": 9,
        "pm2_5_lag_1": 15.0,
        "pm2_5_lag_3": 16.0,
        "pm2_5_lag_24": 18.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_pm2_5" in response.json()