import pandas as pd

DATA_PATH = "data/raw/air_quality.csv"


def validate_data(df):

    required_columns = [
        "time",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    # Check required columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Convert time to datetime
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Check invalid timestamps
    if df["time"].isnull().any():
        raise ValueError("Invalid or missing timestamps found.")

    # Check duplicate timestamps
    if df["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    # Check timestamps are sorted
    if not df["time"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in chronological order.")

    # Check numeric columns
    numeric_columns = required_columns[1:]

    if df[numeric_columns].isnull().any().any():
        raise ValueError("Missing values found in numeric columns.")

    # Check numeric data types
    if not all(
        pd.api.types.is_numeric_dtype(df[column])
        for column in numeric_columns
    ):
        raise ValueError("Non-numeric values found in air-quality columns.")

    # Check for negative values
    if (df[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative values found in air-quality measurements."
        )

    # Check hourly frequency
    time_diff = df["time"].diff().dropna()

    if not (time_diff == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Timestamps are not consistently spaced at 1-hour intervals."
        )

    return True


if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    validate_data(df)

    print("Data validation successful!")
    print(f"Validated records: {len(df)}")
    print(f"Start time: {df['time'].min()}")
    print(f"End time: {df['time'].max()}")