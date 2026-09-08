import mlflow
import mlflow.sklearn
import joblib
import os

MODEL_PATH = "models/linear_regression_pipeline.pkl"

mlflow.set_experiment("air_quality_prediction")

with mlflow.start_run():

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Log model information
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("features", 11)

    # Log evaluation metrics
    mlflow.log_metric("MAE", 0.6371259943718502)
    mlflow.log_metric("RMSE", 0.968568086489315)
    mlflow.log_metric("R2", 0.9495586947829013)

    # Log trained model
    mlflow.sklearn.log_model(
        model,
        name="linear_regression_model"
    )

    print("MLflow run completed successfully!")