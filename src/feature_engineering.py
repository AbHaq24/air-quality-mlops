import pandas as pd


FEATURES = [
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
    "pm2_5_lag_24"
]


def create_features(df):

    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])

    # Time-based features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    # Lag features
    df["pm2_5_lag_1"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_3"] = df["pm2_5"].shift(3)
    df["pm2_5_lag_24"] = df["pm2_5"].shift(24)

    # Remove rows created by lagging
    df = df.dropna().reset_index(drop=True)

    return df


def get_features(df):

    return df[FEATURES]


if __name__ == "__main__":

    DATA_PATH = "data/raw/air_quality.csv"

    df = pd.read_csv(DATA_PATH)

    df = create_features(df)

    X = get_features(df)

    print("Feature engineering successful!")
    print(f"Feature data shape: {X.shape}")
    print("\nFeatures:")
    print(X.columns.tolist())