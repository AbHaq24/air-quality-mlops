import pandas as pd

from src.data_ingestion import clean_data


def test_clean_data():
    data = {
        "time": [
            "2026-01-01 02:00",
            "2026-01-01 01:00",
            "2026-01-01 01:00",
        ],
        "pm2_5": [20.0, 10.0, 10.0],
        "pm10": [30.0, 20.0, 20.0],
        "carbon_monoxide": [200.0, 100.0, 100.0],
        "nitrogen_dioxide": [15.0, 10.0, 10.0],
        "sulphur_dioxide": [2.0, 1.0, 1.0],
        "ozone": [60.0, 50.0, 50.0],
    }

    df = pd.DataFrame(data)

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 2
    assert cleaned_df["time"].is_monotonic_increasing
    assert cleaned_df["time"].is_unique
    assert len(cleaned_df.columns) == 7



    ## Validation test
    
from src.data_validation import validate_data


def test_validate_data():
    data = {
        "time": pd.date_range(
            start="2026-01-01 00:00",
            periods=3,
            freq="h"
        ),
        "pm2_5": [10.0, 11.0, 12.0],
        "pm10": [20.0, 21.0, 22.0],
        "carbon_monoxide": [100.0, 101.0, 102.0],
        "nitrogen_dioxide": [5.0, 6.0, 7.0],
        "sulphur_dioxide": [1.0, 1.0, 1.0],
        "ozone": [50.0, 51.0, 52.0],
    }

    df = pd.DataFrame(data)

    assert validate_data(df) is True

# Feature engg test

from src.feature_engineering import create_features, get_features


def test_feature_engineering():
    data = {
        "time": pd.date_range(
            start="2026-01-01 00:00",
            periods=30,
            freq="h"
        ),
        "pm2_5": range(30),
        "pm10": range(30),
        "carbon_monoxide": range(30),
        "nitrogen_dioxide": range(30),
        "sulphur_dioxide": range(30),
        "ozone": range(30),
    }

    df = pd.DataFrame(data)

    featured_df = create_features(df)
    X = get_features(featured_df)

    assert len(featured_df) == 6
    assert X.shape[1] == 11
    assert list(X.columns) == [
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "hour",
        "day_of_week",
        "month",
        "pm2_5_lag_1",
        "pm2_5_lag_3",
        "pm2_5_lag_24",
    ]