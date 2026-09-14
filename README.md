# Air Quality MLOps

An end-to-end Machine Learning Operations (MLOps) project for predicting **PM2.5 air pollution levels** using air-quality data from the Open-Meteo Air Quality API.

The project demonstrates the complete ML lifecycle from data ingestion and validation to model prediction, experiment tracking, API serving, containerization, automated testing, CI, and cloud deployment.

---

## 1. Problem Statement

Air pollution monitoring requires timely analysis of multiple pollutant measurements. This project builds a machine learning pipeline that uses air-quality measurements and time-based features to predict **PM2.5 concentration**.

The primary goal is to demonstrate how a machine learning model can be integrated into a reproducible and deployable MLOps workflow.

---

## 2. Project Objectives

* Collect air-quality data from an external API.
* Validate and clean incoming data.
* Create features for machine learning.
* Train and use a regression model for PM2.5 prediction.
* Track pipeline runs using MLflow.
* Expose predictions through a FastAPI REST API.
* Containerize the application using Docker.
* Add automated testing with Pytest.
* Maintain code quality using Ruff.
* Implement Continuous Integration using GitHub Actions.
* Deploy the API to the cloud using Render.

---

## 3. Data Source

The project uses the **Open-Meteo Air Quality API**.

Location used:

* Latitude: `12.9716`
* Longitude: `77.5946`

Pollutants collected:

* PM2.5
* PM10
* Carbon Monoxide
* Nitrogen Dioxide
* Sulphur Dioxide
* Ozone

The data is stored in:

```text
data/raw/air_quality.csv
```

The ingestion process supports incremental updates by retrieving recent data and adding only records newer than the latest timestamp in the existing dataset.

---

## 4. MLOps Architecture

```text
Open-Meteo Air Quality API
            │
            ▼
      Data Ingestion
            │
            ▼
      Data Validation
            │
            ▼
    Feature Engineering
            │
            ▼
       ML Model
            │
            ├──────────► MLflow Tracking
            │
            ▼
       FastAPI API
            │
            ▼
         Docker
            │
            ▼
       Render Cloud
```

The development and quality workflow is supported by:

```text
Git → GitHub → GitHub Actions → Tests / Ruff / Docker Build
```

---

## 5. Data Validation

The pipeline validates the incoming dataset before generating predictions.

Validation checks include:

* Required columns are present.
* Timestamps can be parsed correctly.
* Invalid timestamps are rejected.
* Duplicate timestamps are removed/checked.
* Data is sorted chronologically.
* Missing numeric values are handled.
* Negative pollutant values are rejected.

This helps prevent invalid data from entering the downstream ML pipeline.

---

## 6. Feature Engineering

The project creates both pollutant and time-based features.

### Pollutant Features

* PM10
* Carbon Monoxide
* Nitrogen Dioxide
* Sulphur Dioxide
* Ozone

### Time Features

* Hour
* Day of week
* Month

### Lag Features

* PM2.5 lag 1 hour
* PM2.5 lag 3 hours
* PM2.5 lag 24 hours

These features capture recent pollution patterns and temporal behavior.

---

## 7. Machine Learning Model

A **Linear Regression** model was selected for PM2.5 prediction.

The saved model is a Scikit-learn pipeline containing:

```text
StandardScaler
      ↓
LinearRegression
```

The trained model is stored at:

```text
models/linear_regression_pipeline.pkl
```

Earlier model evaluation produced approximately:

| Metric | Result |
| ------ | -----: |
| MAE    | 0.6371 |
| RMSE   | 0.9686 |
| R²     | 0.9496 |

A Random Forest model was also evaluated during development, but Linear Regression provided better results for this dataset.

---

## 8. MLflow Experiment Tracking

MLflow is used to track pipeline runs and prediction statistics.

Experiment:

```text
air_quality_prediction
```

Tracked information includes:

* Pipeline name
* Model type
* Number of features
* Dataset row count
* Mean prediction
* Minimum prediction
* Maximum prediction

The local MLflow tracking database and artifacts are excluded from Git using `.gitignore`.

---

## 9. Project Structure

```text
air-quality-mplos/
│
├── app/
│   └── main.py
│
├── data/
│   └── raw/
│       └── air_quality.csv
│
├── models/
│   └── linear_regression_pipeline.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── orchestration/
│   └── pipeline.py
│
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   └── feature_engineering.py
│
├── tests/
│   └── test_pipeline.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── .gitignore
└── README.md
```

---

## 10. Pipeline Orchestration

The complete pipeline is orchestrated through:

```text
orchestration/pipeline.py
```

The pipeline performs:

```text
1. Data ingestion
2. Data validation
3. Feature engineering
4. Model loading
5. Prediction generation
6. MLflow tracking
```

The trained model is loaded from the saved model artifact rather than retrained during the prediction pipeline.

---

## 11. FastAPI

FastAPI is used to expose the machine learning model as a REST API.

Endpoint:

```text
POST /predict
```

Example input:

```json
{
  "pm10": 30,
  "carbon_monoxide": 200,
  "nitrogen_dioxide": 20,
  "sulphur_dioxide": 5,
  "ozone": 50,
  "hour": 12,
  "day_of_week": 2,
  "month": 9,
  "pm2_5_lag_1": 15,
  "pm2_5_lag_3": 16,
  "pm2_5_lag_24": 18
}
```

Example response:

```json
{
  "predicted_pm2_5": 15.742292739316028
}
```

Interactive API documentation is available through FastAPI Swagger UI at:

```text
/docs
```

---

## 12. Docker

The FastAPI application is containerized using Docker.

The Docker image contains:

* Python runtime
* API dependencies
* FastAPI application
* Trained ML model

The application runs using Uvicorn on port `8000`.

Build locally:

```bash
docker compose build
```

Run locally:

```bash
docker compose up
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 13. Testing

Pytest is used for automated testing of the project.

The project currently contains tests covering important pipeline and API functionality.

Run tests:

```bash
python -m pytest
```

Current test suite:

```text
5 passed
```

---

## 14. Code Quality

Ruff is used for Python linting and code quality checks.

Run:

```bash
python -m ruff check . --exclude notebooks
```

The project passes the Ruff checks.

---

## 15. Continuous Integration

GitHub Actions is used to automatically validate changes pushed to the `main` branch and pull requests.

The CI workflow performs:

```text
Checkout code
      ↓
Set up Python
      ↓
Install dependencies
      ↓
Run Ruff
      ↓
Run MLOps pipeline
      ↓
Run Pytest
      ↓
Build Docker image
```

This helps ensure that code changes do not break the pipeline, tests, or Docker build.

---

## 16. Cloud Deployment

The FastAPI application is deployed using **Render** with Docker.

Deployment flow:

```text
GitHub Repository
        ↓
      Render
        ↓
    Docker Build
        ↓
   FastAPI Container
        ↓
   Public REST API
```

Render is configured for automatic deployment when changes are pushed to the `main` branch.

### Live API

**Swagger UI:**

https://air-quality-mlops-api.onrender.com/docs

---

## 17. Technologies Used

| Technology     | Purpose                    |
| -------------- | -------------------------- |
| Python         | Programming language       |
| Pandas         | Data processing            |
| Scikit-learn   | Machine learning           |
| MLflow         | Experiment tracking        |
| FastAPI        | REST API                   |
| Uvicorn        | API server                 |
| Docker         | Containerization           |
| Docker Compose | Local container management |
| Pytest         | Automated testing          |
| Ruff           | Code quality               |
| Git            | Version control            |
| GitHub         | Source code hosting        |
| GitHub Actions | Continuous Integration     |
| Render         | Cloud deployment           |
| YAML           | CI workflow configuration  |

---

## 18. Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/AbHaq24/air-quality-mlops.git
cd air-quality-mlops
```

### Create virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### Run the pipeline

```bash
python orchestration/pipeline.py
```

### Run tests

```bash
python -m pytest
```

### Run the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### Run with Docker

```bash
docker compose up --build
```

---

## 19. Future Improvements

Potential future improvements include:

* Automated scheduled data ingestion.
* Automated model retraining when new data becomes available.
* Model performance monitoring.
* Data drift detection.
* Model version management.
* CI/CD deployment gates.
* Cloud-based MLflow tracking.
* Monitoring and alerting for the deployed API.

---

## 20. Conclusion

This project demonstrates an end-to-end MLOps workflow for an air-quality prediction application.

The solution integrates data ingestion, validation, feature engineering, machine learning, experiment tracking, API serving, testing, containerization, Continuous Integration, and cloud deployment into a single reproducible workflow.

The project focuses on demonstrating how a machine learning model can move from development to a deployable production-style API using modern MLOps practices.
