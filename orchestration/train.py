import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import create_features, get_features


DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "air_quality.csv",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "linear_regression_pipeline.pkl",
)


def train_model():
    print("=" * 50)
    print("AIR QUALITY MODEL TRAINING")
    print("=" * 50)

    # Load data
    print("\n[1/4] Loading data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # Create features
    print("\n[2/4] Creating features...")

    feature_df = create_features(df)
    X = get_features(feature_df)
    y = feature_df["pm2_5"]

    print(f"Feature shape: {X.shape}")

    # Train model
    print("\n[3/4] Training Linear Regression model...")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X, y)

    print("Model training successful!")

    # Save model
    print("\n[4/4] Saving model...")

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("MODEL TRAINING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    train_model()