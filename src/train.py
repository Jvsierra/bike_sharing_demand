"""Train the LightGBM bike-sharing-demand model and log it with mlflow.

Usage:
    python -m src.train
    python -m src.train --train-path data/train.csv --n-splits 5

The full preprocessing (one-hot encoding of season/weather) plus the LGBM
regressor are wrapped in a single sklearn Pipeline so mlflow logs and
versions them together as one model artifact. Inference (predict.py) loads
that same artifact back, so training and inference can never drift apart.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CATEGORICAL_FEATURES, prepare_train_features

REPO_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42
LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, max_depth=5, num_leaves=15,
    min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1,
)


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LGBMRegressor(**LGBM_PARAMS)),
    ])


def compute_metrics(y_true_count, y_pred_log):
    count_pred = np.maximum(np.expm1(y_pred_log), 0)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true_count), np.log1p(count_pred)))
    mae = mean_absolute_error(y_true_count, count_pred)
    r2 = r2_score(y_true_count, count_pred)
    return {"RMSLE": rmsle, "MAE": mae, "R2": r2}


def cross_validate(X, y_log, y_count, n_splits):
    """Walk-forward TimeSeriesSplit CV, mirroring the experimentation notebook."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        pipeline = build_pipeline()
        pipeline.fit(X.iloc[train_idx], y_log.iloc[train_idx])
        test_pred_log = pipeline.predict(X.iloc[test_idx])
        metrics = compute_metrics(y_count.iloc[test_idx], test_pred_log)
        fold_metrics.append(metrics)
        for name, value in metrics.items():
            mlflow.log_metric(f"cv_{name}", value, step=fold)
    mean_metrics = {
        name: float(np.mean([fold[name] for fold in fold_metrics]))
        for name in fold_metrics[0]
    }
    return mean_metrics


def train(train_path, n_splits, experiment_name, tracking_uri):
    """Run CV + fit the final pipeline, log everything to mlflow, return the run_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X, y_log, y_count = prepare_train_features(train_path)

    with mlflow.start_run() as run:
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("cv_strategy", "TimeSeriesSplit (walk-forward)")
        mlflow.log_param("target_transform", "log1p")

        mean_metrics = cross_validate(X, y_log, y_count, n_splits)
        for name, value in mean_metrics.items():
            mlflow.log_metric(f"cv_mean_{name}", value)
        print(f"CV mean metrics ({n_splits}-fold, TimeSeriesSplit): {mean_metrics}")

        final_pipeline = build_pipeline()
        final_pipeline.fit(X, y_log)

        signature = infer_signature(X, final_pipeline.predict(X))
        mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X.head(3),
        )
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(REPO_ROOT / "data" / "train.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--experiment-name", default="bike-sharing-demand")
    parser.add_argument("--tracking-uri", default=f"file:{REPO_ROOT / 'mlruns'}")
    args = parser.parse_args()

    run_id = train(args.train_path, args.n_splits, args.experiment_name, args.tracking_uri)
    print(f"Run ID: {run_id}")
    print(f"Model logged at: runs:/{run_id}/model")


if __name__ == "__main__":
    main()
