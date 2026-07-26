# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Predicts hourly bike-sharing demand (Kaggle "Bike Sharing Demand" dataset). A LightGBM
regressor is trained on `data/train.csv` and scored against `data/test.csv` to produce a
Kaggle-format submission at `data/submissions.csv`.

## Environment

A `.venv` already exists in the repo root (Windows venv layout: `.venv/Scripts/python.exe`,
not `.venv/bin/`). Use it directly rather than creating a new one:

```
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Note `requirements.txt` covers Jupyter/notebook and general data-science tooling but not
every runtime dependency actually imported by `src/` (e.g. `mlflow`, `lightgbm`, `flask`,
`waitress` were installed separately — see `.claude/settings.json` for the exact install
commands used). If a module is missing, install it into `.venv` rather than assuming it's
covered by `requirements.txt`.

## Commands

Run everything through the venv's Python using `-m` so package-relative imports (`src.*`)
resolve correctly:

```
./.venv/Scripts/python.exe -m src.train              # train + log a model to mlflow
./.venv/Scripts/python.exe -m src.predict             # score data/test.csv with the latest run
./.venv/Scripts/python.exe -m src.pipeline            # train then predict in one go (a fresh run_id feeds predict directly)
./.venv/Scripts/python.exe -m mlflow ui --port 5551   # inspect experiment runs/metrics
```

Each of `train.py`, `predict.py`, and `pipeline.py` exposes CLI args (`--train-path`,
`--test-path`, `--output-path`, `--n-splits`, `--experiment-name`, `--tracking-uri`,
`--run-id`) with sensible defaults pointing at `data/` and `mlruns/` under the repo root —
check `argparse` setup in each file before assuming a flag doesn't exist.

There is no test suite and no lint/format tooling configured in this repo.

## Architecture

Three-module `src/` package, meant to be run as scripts (`python -m src.<name>`) rather
than imported piecemeal:

- **`src/features.py`** — shared feature engineering, deliberately kept as plain pandas
  (not part of the sklearn `Pipeline`) because it operates on the raw datetime column and,
  for training only, needs to reindex/impute rows before a target exists.
  - `complete_missing_train_hours`: `train.csv` only ever contains days 1–19 of each
    month; gaps get forward-filled but reindexing stays within each month's 1–19 window
    rather than the global min/max datetime, or it would fabricate rows for days that
    never existed in the source data.
  - `prepare_test_features` deliberately does **not** run gap-filling — `test.csv`'s
    missing hours are genuine Kaggle-excluded rows, and predictions must line up 1:1 with
    the rows actually present.
  - The target is modeled in log space (`np.log1p(count)`); inverse-transform with
    `np.expm1` and clip at 0 when producing real predictions (see `predict.py`).

- **`src/train.py`** — builds an sklearn `Pipeline` (one-hot encode
  `season`/`weather` via `ColumnTransformer`, then `LGBMRegressor`) so mlflow logs and
  versions preprocessing + model as a single artifact — this is why one-hot encoding
  lives here and not in `features.py`. Cross-validation uses `TimeSeriesSplit`
  (walk-forward, not random/K-fold) to mirror the notebook experiments and respect the
  data's temporal ordering. `train()` logs params/CV metrics/final model to mlflow and
  returns the run's `run_id`.

- **`src/predict.py`** — loads a model back via its `runs:/<run_id>/model` URI (no
  Model Registry involved). Without `--run-id`, resolves the most recent `FINISHED` run
  of the experiment. Writes a two-column (`datetime`, `count`) CSV submission.

- **`src/pipeline.py`** — chains `train.train()` → `predict.predict()`, passing the
  `run_id` explicitly between them instead of having `predict` re-resolve "latest" — avoids
  ambiguity if another run gets logged concurrently.

mlflow tracking store is local file-based (`file:<repo_root>/mlruns`), experiment name
defaults to `bike-sharing-demand`.

`notebooks/EDA.ipynb` and `notebooks/Models Experiments.ipynb` hold the exploratory work
that `src/` formalizes — check them for rationale behind modeling choices before changing
feature engineering or model params.
