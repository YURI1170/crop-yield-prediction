"""Train and evaluate the reproducible crop-yield regression workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

try:
    from src.inference import FEATURE_COLUMNS
except ModuleNotFoundError:  # Supports `python src/train.py` from the project root.
    from inference import FEATURE_COLUMNS


def inverse_target(values, target_offset: float):
    return np.exp(values) - target_offset


def metrics_for_predictions(target_transformed, predictions_transformed, target_offset):
    target_original = inverse_target(target_transformed, target_offset)
    predictions_original = inverse_target(predictions_transformed, target_offset)
    return {
        "transformed_target": {
            "mae": float(mean_absolute_error(target_transformed, predictions_transformed)),
            "rmse": float(mean_squared_error(target_transformed, predictions_transformed) ** 0.5),
            "r2": float(r2_score(target_transformed, predictions_transformed)),
        },
        "original_target": {
            "mae": float(mean_absolute_error(target_original, predictions_original)),
            "rmse": float(mean_squared_error(target_original, predictions_original) ** 0.5),
            "r2": float(r2_score(target_original, predictions_original)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yield_df.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    data = pd.read_csv(args.data).drop(columns=["Unnamed: 0"], errors="ignore")
    before_deduplication = len(data)
    data = data.drop_duplicates().copy()
    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values must be handled before training: {missing}")

    target_offset = float(data["hg/ha_yield"].min())
    target = np.log(data.pop("hg/ha_yield") + target_offset)
    data["pesticides_tonnes_log"] = np.log1p(data.pop("pesticides_tonnes"))
    features = data[FEATURE_COLUMNS]

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=0.30, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, random_state=42
    )

    categorical = ["Area", "Item"]
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_val_transformed = preprocessor.transform(x_val)
    x_test_transformed = preprocessor.transform(x_test)

    model = XGBRegressor(
        n_estimators=26,
        max_depth=9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    model.fit(x_train_transformed, y_train)

    val_predictions = model.predict(x_val_transformed)
    test_predictions = model.predict(x_test_transformed)
    metrics = {
        "validation": metrics_for_predictions(y_val, val_predictions, target_offset),
        "test": metrics_for_predictions(y_test, test_predictions, target_offset),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, args.output / "preprocessor.joblib")
    joblib.dump(model, args.output / "xgboost_model.joblib")
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": categorical,
        "target": "hg/ha_yield",
        "target_transform": "log(yield + target_offset)",
        "target_offset": target_offset,
        "random_state": 42,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "model": {"type": "XGBRegressor", "n_estimators": 26, "max_depth": 9},
        "runtime": {"xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__},
        "rows_before_deduplication": before_deduplication,
        "rows_after_deduplication": len(data),
        "metrics": metrics,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
