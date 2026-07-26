"""Generate the chart images embedded in README.md.

One-off documentation utility, not part of the training/inference pipeline.
Re-run whenever the model comparison numbers or feature set change:

    ./.venv/Scripts/python.exe -m scripts.generate_readme_assets
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.features import prepare_train_features
from src.train import build_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"

# From notebooks/Models Experiments.ipynb, section 9 (mean metrics across CV test folds).
MODEL_COMPARISON = {
    "Random Baseline": 2.064,
    "Linear Regression": 1.080,
    "Decision Tree": 0.698,
    "Random Forest": 0.549,
    "XGBoost": 0.455,
    "LightGBM": 0.456,
}


def plot_model_comparison():
    models = list(MODEL_COMPARISON.keys())
    rmsle = list(MODEL_COMPARISON.values())
    colors = ["#4c72b0" if m != "LightGBM" else "#dd8452" for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(models, rmsle, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test RMSLE (lower is better)")
    ax.set_title("Model comparison — walk-forward CV")
    for bar, value in zip(bars, rmsle):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted():
    X, y_log, y_count = prepare_train_features(REPO_ROOT / "data" / "train.csv")
    train_idx, test_idx = list(TimeSeriesSplit(n_splits=5).split(X))[-1]

    pipeline = build_pipeline()
    pipeline.fit(X.iloc[train_idx], y_log.iloc[train_idx])
    pred_count = np.maximum(np.expm1(pipeline.predict(X.iloc[test_idx])), 0)

    window = slice(0, 24 * 14)  # first two weeks of the held-out fold
    actual = y_count.iloc[test_idx].to_numpy()[window]
    predicted = pred_count[window]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label="Actual", color="#4c72b0", linewidth=1.2)
    ax.plot(predicted, label="Predicted (LightGBM)", color="#dd8452", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Hours into held-out fold")
    ax.set_ylabel("Bike rentals (count)")
    ax.set_title("Actual vs. predicted demand — held-out fold, first two weeks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    plot_model_comparison()
    plot_actual_vs_predicted()
    print(f"Wrote assets to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
