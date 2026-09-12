"""Shared artifact loading and prediction logic for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes_log",
]


def load_data(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "data" / "yield_df.csv")


def load_artifacts(project_root: Path):
    artifact_dir = project_root / "artifacts"
    required = [
        artifact_dir / "preprocessor.joblib",
        artifact_dir / "xgboost_model.joblib",
        artifact_dir / "metadata.json",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run `python src/train.py --data data/yield_df.csv --output artifacts` first."
        )
    with (artifact_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("The saved artifact feature schema does not match the application schema.")
    return joblib.load(required[0]), joblib.load(required[1]), metadata


def make_input_frame(
    area: str,
    item: str,
    year: int,
    rainfall: float,
    temperature: float,
    pesticides: float,
) -> pd.DataFrame:
    values = [rainfall, temperature, pesticides]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("All numeric inputs must be finite numbers.")
    if rainfall < 0 or pesticides < 0:
        raise ValueError("Rainfall and pesticide use cannot be negative.")
    return pd.DataFrame(
        [
            {
                "Area": area,
                "Item": item,
                "Year": int(year),
                "average_rain_fall_mm_per_year": float(rainfall),
                "avg_temp": float(temperature),
                "pesticides_tonnes_log": float(np.log1p(pesticides)),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_yield(input_frame: pd.DataFrame, preprocessor, model, metadata) -> float:
    transformed = preprocessor.transform(input_frame[FEATURE_COLUMNS])
    transformed_prediction = float(model.predict(transformed)[0])
    target_offset = float(metadata["target_offset"])
    return max(0.0, float(np.exp(transformed_prediction) - target_offset))
