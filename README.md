# Crop Yield Prediction with XGBoost

This project is a small regression demo for estimating historical crop yield from area, crop type, year, rainfall, temperature, and pesticide usage. The workflow uses a saved preprocessing pipeline and an XGBoost regressor, with a Streamlit interface for interactive prediction.

## Objective

The objective is to predict crop yield in hectograms per hectare (`hg/ha_yield`) from structured agricultural and environmental features. The model is intended as a local machine-learning exercise and should not be treated as a production agricultural forecasting system.

## Dataset

The project reads the dataset from `data/yield_df.csv`. The local CSV is intentionally excluded from Git until its exact source and redistribution terms are verified. A fresh clone therefore requires you to obtain the permitted dataset copy and place it at that path before training.

The file contains rows for agricultural yield records with these fields:

- `Area`
- `Item`
- `Year`
- `hg/ha_yield`
- `average_rain_fall_mm_per_year`
- `pesticides_tonnes`
- `avg_temp`

The project removes the unused index column and exact duplicate rows before fitting the model. The dataset is historical and aggregated, and the exact citation and redistribution terms should be confirmed before publishing a public copy of the CSV.

## Preprocessing

The training workflow follows this sequence:

1. Read the CSV and drop the unused index column if present.
2. Drop duplicate rows.
3. Define the feature set:
   - `Area`
   - `Item`
   - `Year`
   - `average_rain_fall_mm_per_year`
   - `avg_temp`
   - `pesticides_tonnes_log`
4. Create the target as a log-transformed value using an offset derived from the minimum observed yield.
5. Split the data into train/validation/test sets using `train_test_split(..., random_state=42)`.
6. Fit a `ColumnTransformer` with an `OrdinalEncoder` for the categorical columns `Area` and `Item`.
7. Transform all splits with the same saved preprocessor before model training and prediction.

The application uses the same schema and the same saved preprocessor from `artifacts/preprocessor.joblib`.

## XGBoost model

The model is an `XGBRegressor` configured as:

- `n_estimators=26`
- `max_depth=9`
- `random_state=42`
- `objective="reg:squarederror"`

The training code saves the fitted regressor to `artifacts/xgboost_model.joblib` and stores the schema and metrics in `artifacts/metadata.json`.

## Target transformation

The original target is transformed with:

```python
target_offset = float(data["hg/ha_yield"].min())
target = np.log(data.pop("hg/ha_yield") + target_offset)
```

This means the model is trained on `log(yield + offset)`, not on the raw `hg/ha_yield` scale. The stored offset is `50.0` for the supplied dataset. Inference reverses the transform with:

```python
prediction_original = np.exp(model_output) - target_offset
```

The app reads the offset from the saved metadata and applies this inversion before showing the result.

## Training

From the project root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py --data data/yield_df.csv --output artifacts
```

This writes the following files to `artifacts/`. The generated model binaries are also excluded from Git by default; regenerate them locally before running the application from a fresh clone.

- `preprocessor.joblib`
- `xgboost_model.joblib`
- `metadata.json`

## Prediction workflow

The inference logic is centralized in `src/inference.py` and is shared by both the dashboard and the page script.

The prediction flow is:

1. Validate numeric user inputs.
2. Build a one-row DataFrame with the exact expected feature columns.
3. Apply the saved `OrdinalEncoder` preprocessor.
4. Run the saved XGBoost model.
5. Invert the target transform using the saved `target_offset`.
6. Return the final yield estimate in `hg/ha`.

The app also clamps the final predicted yield to a non-negative value before display.

## Streamlit usage

Run the app from the project root:

```bash
streamlit run Dashboard.py
```

The dashboard exposes a form with:

- `Area`
- `Crop`
- `Year`
- rainfall
- temperature
- pesticide usage

It shows the predicted yield and a historical context chart for the selected area and crop.

## Evaluation

The training script records MAE, RMSE, and R² for both the transformed target and the original target scale. The real evaluation generated from this project after duplicate removal is:

| Split | Target scale | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Validation | Original `hg/ha` | 6,651.23 | 14,634.73 | 0.9714 |
| Test | Original `hg/ha` | 6,707.99 | 14,644.09 | 0.9719 |

These metrics were produced by the checked-in training script using a `random_state=42` split and the dataset in `data/yield_df.csv`.

## Limitations

- The dataset is historical and aggregated, not a real-time operational monitoring source.
- Random splitting does not test future-year or geographic generalization.
- The model does not imply causal relationships.
- `OrdinalEncoder` is a simple encoding choice for tree-based models; other encodings could be evaluated.
- The exact external dataset source, version, and license were not preserved in the recovered project files, so publication checks are still required.

## Future improvements

Potential next steps include:

- temporal holdout evaluation
- geographic holdout evaluation
- uncertainty estimates
- feature importance and explainability analysis
- comparison against alternative models
- stronger source and licensing verification before public release

## Sources and licenses

The local CSV matches the [Kaggle Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset), which credits [FAO](https://www.fao.org/home/en/) and [World Bank](https://data.worldbank.org/) inputs. The Kaggle page lists **World Bank Dataset Terms of Use**. The raw CSV remains excluded from Git because the exact version and redistribution rights for the recovered copy must still be confirmed. See `DATA_SOURCES.md` before redistributing data or model artifacts.

## Author and license

This project is authored and maintained by Abdellah Guemmah.

This repository is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.
