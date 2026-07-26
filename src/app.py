"""Streamlit page showing the model's predictions for data/test.csv.

Single-page app: reads the submission file produced by predict.py/pipeline.py
and displays it as one table with human-readable column names.

Usage:
    ./.venv/Scripts/python.exe -m streamlit run src/app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_PATH = REPO_ROOT / "data" / "submissions.csv"

st.set_page_config(page_title="Bike Sharing Demand - Predictions")
st.title("Bike Sharing Demand - Predictions")

if not SUBMISSIONS_PATH.exists():
    st.error(
        f"No predictions found at `{SUBMISSIONS_PATH}`. "
        "Run `python -m src.pipeline` (or `python -m src.predict`) first."
    )
else:
    predictions = pd.read_csv(SUBMISSIONS_PATH, parse_dates=["datetime"])
    predictions["count"] = predictions["count"].round().astype(int)
    predictions = predictions.rename(columns={
        "datetime": "Date & Time",
        "count": "Predicted Bike Rentals",
    })
    st.dataframe(predictions, hide_index=True)
