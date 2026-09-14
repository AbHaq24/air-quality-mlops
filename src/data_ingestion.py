import os

import pandas as pd
import requests

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

PARAMS = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "timezone": "auto",
    "past_days": 2
}

DATA_PATH = "data/raw/air_quality.csv"

NUMERIC_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
]


def fetch_air_quality_data():

    response = requests.get(
        URL,
        params=PARAMS,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return pd.DataFrame(data["hourly"])


def clean_data(df):

    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])

    # Remove duplicate timestamps
    df = df.drop_duplicates(
        subset=["time"]
    )

    # Sort chronologically
    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # Handle missing pollutant values
    df[NUMERIC_COLUMNS] = (
        df[NUMERIC_COLUMNS]
        .interpolate(method="linear")
    )

    return df


def update_air_quality_data():

    # -------------------------------------------------
    # Initial dataset creation
    # -------------------------------------------------

    if not os.path.exists(DATA_PATH):

        print("Existing dataset not found.")
        print("Creating initial dataset...")

        new_data = fetch_air_quality_data()

        new_data = clean_data(new_data)

        new_data.to_csv(
            DATA_PATH,
            index=False
        )

        print(
            f"Initial dataset created: "
            f"{new_data.shape}"
        )

        return new_data

    # -------------------------------------------------
    # Load existing dataset
    # -------------------------------------------------

    existing_data = pd.read_csv(DATA_PATH)

    existing_data["time"] = pd.to_datetime(
        existing_data["time"]
    )

    latest_time = existing_data["time"].max()

    print(
        f"Latest existing timestamp: "
        f"{latest_time}"
    )

    # -------------------------------------------------
    # Fetch recent API data
    # -------------------------------------------------

    new_data = fetch_air_quality_data()

    new_data["time"] = pd.to_datetime(
        new_data["time"]
    )

    # Keep only records newer than existing data
    new_data = new_data[
        new_data["time"] > latest_time
    ]

    print(
        f"New records found: "
        f"{len(new_data)}"
    )

    # -------------------------------------------------
    # Append new records
    # -------------------------------------------------

    if len(new_data) > 0:

        existing_data = pd.concat(
            [existing_data, new_data],
            ignore_index=True
        )

        print("New data appended.")

    else:

        print(
            "No new records available."
        )

    # -------------------------------------------------
    # Clean complete dataset
    # -------------------------------------------------

    updated_data = clean_data(
        existing_data
    )

    # -------------------------------------------------
    # Save updated dataset
    # -------------------------------------------------

    updated_data.to_csv(
        DATA_PATH,
        index=False
    )

    print(
        f"Dataset saved successfully: "
        f"{updated_data.shape}"
    )

    return updated_data


if __name__ == "__main__":

    df = update_air_quality_data()

    print("\nFinal dataset:")
    print(f"Rows: {len(df)}")
    print(f"Start: {df['time'].min()}")
    print(f"End: {df['time'].max()}")