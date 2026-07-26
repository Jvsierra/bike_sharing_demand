"""Score test.csv with the LightGBM model logged by train.py and write a
Kaggle-format submission.

By default, loads the most recent successful run of the training experiment
from the mlflow tracking store (no Model Registry involved: the run's own
"model" artifact is loaded directly via its runs:/ URI). Pass --run-id to
pin a specific run instead.

Usage:
    python -m src.predict
    python -m src.predict --run-id <run_id> --test-path data/test.csv
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from src.features import prepare_test_features

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_latest_run_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise MlflowException(f"Experiment '{experiment_name}' not found. Run train.py first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise MlflowException(f"No finished runs found for experiment '{experiment_name}'.")
    return runs.iloc[0]["run_id"]


def predict(test_path, output_path, experiment_name, tracking_uri, run_id=None):
    """Score test_path with the given (or latest finished) run's model, write output_path, return the run_id used."""
    mlflow.set_tracking_uri(tracking_uri)

    run_id = run_id or resolve_latest_run_id(experiment_name)
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    X, datetimes = prepare_test_features(test_path)
    pred_log = model.predict(X)
    count_pred = np.maximum(np.expm1(pred_log), 0)

    submission = pd.DataFrame({"datetime": datetimes, "count": count_pred})
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} predictions to {output_path} (run {run_id})")
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", default=str(REPO_ROOT / "data" / "test.csv"))
    parser.add_argument("--output-path", default=str(REPO_ROOT / "data" / "submissions.csv"))
    parser.add_argument("--experiment-name", default="bike-sharing-demand")
    parser.add_argument("--tracking-uri", default=f"file:{REPO_ROOT / 'mlruns'}")
    parser.add_argument("--run-id", default=None, help="Pin a specific mlflow run instead of using the latest one.")
    args = parser.parse_args()

    predict(args.test_path, args.output_path, args.experiment_name, args.tracking_uri, run_id=args.run_id)


if __name__ == "__main__":
    main()
