"""End-to-end run: train the model, then score test.csv with the model just trained.

Chains train.train() and predict.predict() directly, passing the resulting
run_id explicitly rather than having predict.py re-resolve the "latest"
finished run - avoids ambiguity if another run gets logged concurrently.

Usage:
    python -m src.pipeline
    python -m src.pipeline --n-splits 10 --output-path data/submissions.csv
"""

import argparse
from pathlib import Path

from src.predict import predict
from src.train import train

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(REPO_ROOT / "data" / "train.csv"))
    parser.add_argument("--test-path", default=str(REPO_ROOT / "data" / "test.csv"))
    parser.add_argument("--output-path", default=str(REPO_ROOT / "data" / "submissions.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--experiment-name", default="bike-sharing-demand")
    parser.add_argument("--tracking-uri", default=f"file:{REPO_ROOT / 'mlruns'}")
    args = parser.parse_args()

    print("=== Training ===")
    run_id = train(args.train_path, args.n_splits, args.experiment_name, args.tracking_uri)
    print(f"Run ID: {run_id}")
    print(f"Model logged at: runs:/{run_id}/model")

    print("=== Predicting ===")
    predict(args.test_path, args.output_path, args.experiment_name, args.tracking_uri, run_id=run_id)


if __name__ == "__main__":
    main()
