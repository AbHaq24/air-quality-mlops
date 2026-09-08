import sys
import os
import joblib
import mlflow
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion import update_air_quality_data
from src.data_validation import validate_data
from src.feature_engineering import create_features, get_features


# Paths
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "air_quality.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "linear_regression_pipeline.pkl"
)


def run_pipeline():

    print("=" * 50)
    print("AIR QUALITY MLOPS PIPELINE")
    print("=" * 50)

    # -------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------

    print("\n[1/6] Updating data...")

    df = update_air_quality_data()

    print(f"Dataset shape: {df.shape}")

    # -------------------------------------------------
    # 2. DATA VALIDATION
    # -------------------------------------------------

    print("\n[2/6] Validating data...")

    validate_data(df)

    print("Data validation successful!")

    # -------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------

    print("\n[3/6] Creating features...")

    feature_df = create_features(df)

    X = get_features(feature_df)

    y = feature_df["pm2_5"]

    print(f"Feature shape: {X.shape}")

    # -------------------------------------------------
    # 4. LOAD MODEL
    # -------------------------------------------------

    print("\n[4/6] Loading model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

    # -------------------------------------------------
    # 5. PREDICTION
    # -------------------------------------------------

    print("\n[5/6] Generating predictions...")

    predictions = model.predict(X)

    feature_df["predicted_pm2_5"] = predictions

    print("Predictions generated successfully!")

    print("\nSample predictions:")

    print(
        feature_df[
            ["time", "pm2_5", "predicted_pm2_5"]
        ].tail()
    )

    # -------------------------------------------------
    # 6. MLFLOW TRACKING
    # -------------------------------------------------

    print("\n[6/6] Logging pipeline run to MLflow...")

    mlflow.set_tracking_uri(
        "sqlite:///E:/Praxis PGDDS/Term 2/Mlops/air-quality-mplos/mlflow.db"
    )

    mlflow.set_experiment(
        "air_quality_prediction"
    )

    with mlflow.start_run():

        mlflow.log_param(
            "pipeline",
            "air_quality_pipeline"
        )

        mlflow.log_param(
            "model_type",
            "Linear Regression"
        )

        mlflow.log_param(
            "feature_count",
            len(X.columns)
        )

        mlflow.log_param(
            "dataset_rows",
            len(df)
        )

        mlflow.log_metric(
            "prediction_mean",
            float(predictions.mean())
        )

        mlflow.log_metric(
            "prediction_min",
            float(predictions.min())
        )

        mlflow.log_metric(
            "prediction_max",
            float(predictions.max())
        )

        print("MLflow logging successful!")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()