"""Shared feature engineering for training and inference.

Kept as plain pandas transforms (not part of the sklearn Pipeline) because
they operate on the raw datetime column and, for training only, need to
reindex/impute rows before a target column exists. The one-hot encoding of
categorical columns *is* part of the model pipeline (see `build_pipeline` in
train.py) so it is versioned and loaded together with the model by mlflow.
"""

import numpy as np
import pandas as pd

SEASON_LABELS = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_LABELS = {
    1: "Clear/Few Clouds",
    2: "Mist/Cloudy",
    3: "Light Rain/Snow",
    4: "Heavy Rain/Snow",
}

CATEGORICAL_FEATURES = ["season", "weather"]
NUMERICAL_FEATURES = [
    "temp", "atemp", "day_of_week", "humidity", "hour", "month", "year",
    "windspeed", "workingday", "holiday",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_raw(path):
    return pd.read_csv(path, parse_dates=["datetime"])


def complete_missing_train_hours(df):
    """Fill gaps in train.csv's hourly series (days 1-19 of each month only).

    Train only ever contains days 1-19 of each month, so reindexing must stay
    within that per-month window instead of the global min/max datetime,
    which would fabricate rows for days 20-end of month that never existed.
    Missing rows are forward-filled from the previous hour.
    """
    original_dtypes = df.drop(columns="datetime").dtypes
    df = df.set_index("datetime").sort_index()

    months_present = df.index.to_period("M").unique()
    full_range = pd.DatetimeIndex(sorted(
        ts
        for month in months_present
        for ts in pd.date_range(month.start_time, periods=19 * 24, freq="h")
    ))
    df = df.reindex(full_range).ffill()
    df.index.name = "datetime"
    return df.reset_index().astype(original_dtypes.to_dict())


def add_datetime_features(df):
    df = df.copy()
    df["season"] = df["season"].map(SEASON_LABELS)
    df["weather"] = df["weather"].map(WEATHER_LABELS)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["day_of_week"] = df["datetime"].dt.dayofweek
    return df


def prepare_train_features(train_path):
    """Read train.csv and return (X, y_log, y_count) ready for the pipeline."""
    df = load_raw(train_path)
    df = complete_missing_train_hours(df)
    df = df.drop(columns=["casual", "registered"])
    df = add_datetime_features(df)

    X = df[FEATURE_COLUMNS]
    y_count = df["count"]
    y_log = pd.Series(np.log1p(y_count), index=y_count.index, name="count_log")
    return X, y_log, y_count


def prepare_test_features(test_path):
    """Read test.csv and return (X, datetime) ready for the pipeline.

    No missing-hour completion here: the gaps in test.csv are genuine rows
    Kaggle excludes from the test set, not data-quality gaps to be filled in,
    and predictions must line up 1:1 with the rows actually present.
    """
    df = load_raw(test_path)
    df = add_datetime_features(df)
    return df[FEATURE_COLUMNS], df["datetime"]
