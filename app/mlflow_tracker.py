"""
MLflow experiment tracking, shared by the live API (per-query runs) and
scripts/run_evaluation.py (offline evaluation sweeps). Uses a SQLite-backed
tracking store — MLflow's plain filesystem store is in maintenance mode as
of MLflow 2.x, so this is the currently-recommended lightweight setup.
"""

import mlflow

from app.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)


def start_run(run_name: str | None = None):
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict) -> None:
    mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})


def active_run_id() -> str | None:
    run = mlflow.active_run()
    return run.info.run_id if run else None
