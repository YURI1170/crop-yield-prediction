# Interview Notes

## What is the project?

This project is a supervised learning regression workflow that estimates historical crop yield from structured agricultural and environmental inputs. The checked-in application uses a saved preprocessing pipeline and a trained XGBoost regressor, with a Streamlit interface for user input.

## What is supervised learning?

Supervised learning trains a model on examples that contain both inputs and a known target. Here, the inputs are features such as area, crop type, year, rainfall, temperature, and pesticide usage, and the target is the observed crop yield in `hg/ha_yield`.

## What is regression?

Regression predicts a continuous numeric value rather than a class label. In this project, the model predicts a numeric yield value and then converts it back to the original yield scale before presentation.

## Why XGBoost?

XGBoost is a gradient-boosted tree algorithm that performs well on structured tabular data with nonlinear relationships and interactions. It is used here as the model family for the crop-yield baseline.

## Why preprocessing?

The project must apply the same feature ordering and category encoding at prediction time that it used during training. The shared functions in `src/inference.py` enforce that contract so the saved preprocessor and model are used consistently.

## Why categorical encoding?

`Area` and `Item` are text categories. The training code uses `OrdinalEncoder` on those columns and stores the fitted encoder in `artifacts/preprocessor.joblib`. Unknown categories are encoded as `-1`, which is a simple and compatible approach for the current tree-based model.

## Why train/validation/test splits?

The dataset is split into training, validation, and test subsets. Training fits the model, validation helps monitor performance during development, and the test split provides the final held-out estimate. The current split uses `random_state=42` and therefore evaluates model performance under a random partition, not a true future-year or geographic generalization test.

## Why transform the target?

The target is transformed with `log(yield + target_offset)`, where `target_offset` is the minimum observed yield value in the dataset. This reduces skewness and helps the model learn on a more manageable scale. In this project, `target_offset` is stored in `artifacts/metadata.json` and equals `50.0`.

## How is the target transformation inverted?

The prediction workflow applies the inverse transform:

```python
np.exp(model_output) - target_offset
```

This is the correct inverse of `log(y + offset)`. The application does not hard-code the offset separately; it reads the saved value from metadata and performs the inverse transform before displaying the result.

## What are MAE, RMSE, and R²?

- MAE (Mean Absolute Error): the average absolute difference between actual and predicted yields.
- RMSE (Root Mean Squared Error): the square root of the average squared error. It penalizes large mistakes more strongly than MAE.
- R² (R-squared): measures how much better the model is than predicting the mean target value. Values closer to 1 indicate stronger explanatory power on the selected split.

For this project, the training script records these metrics in both transformed-target and original-target units.

## How does Streamlit inference work?

The user enters area, crop, year, rainfall, temperature, and pesticide data in the app. The app builds a one-row DataFrame using the exact expected feature names, passes it through the saved preprocessor, runs the saved XGBoost model, inverts the target transformation, and displays a yield estimate in `hg/ha`.

## What did I personally implement?

This project was developed by Abdellah Guemmah. The verified work in this repository includes the feature schema, training script, saved model artifacts, Streamlit app, and supporting documentation.

## Limitations

The dataset is historical and aggregated. Random splitting can overestimate performance when similar rows appear across train and test sets, and the model does not establish causal effects. The project also does not report uncertainty intervals, and the exact external source and license information were not captured in the recovered project files.

## What would you improve?

The next improvements would be temporal and geographic evaluation, uncertainty estimates, crop- or region-specific error analysis, comparison with alternative encodings or models, and formal verification of dataset source and licensing terms.

## Research direction

This project is a useful baseline for structured agricultural prediction, but it is not a finished operational forecasting system. A stronger extension would include holdout evaluation by time and region, explainability, and a more explicit validation plan before any public or business-facing use.
