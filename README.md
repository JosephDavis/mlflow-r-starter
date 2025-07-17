# MLflow R Starter

A Docker-based starter for MLflow experiment tracking with R. Uses the iris dataset and random forest as a demonstration - replace with your own data and models.

## Quick Start

```bash
docker-compose up
```

Open http://localhost:5001 to view MLflow UI.

## Structure

- `R/mlflow_example.R` - Main R script with MLflow integration
- `docker-compose.yml` - MLflow server + R service
- `docker/r-runtime/Dockerfile` - R environment

## Customize

Replace the Iris dataset and Random Forest model in `mlflow_example.R` with your data and algorithm. Update the TODO comments throughout the file.

## Notes

There are some limitations to logging models with R around model signatures: https://github.com/mlflow/mlflow/issues/4462

