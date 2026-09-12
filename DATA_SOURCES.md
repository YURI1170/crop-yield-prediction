# Data Sources and Publication Checks

## Dataset

The local `yield_df.csv` matches the schema, row structure, and sample values of the [Kaggle Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset), including `Area`, `Item`, `Year`, `hg/ha_yield`, rainfall, pesticides, and temperature. The Kaggle page credits the underlying public data to the [Food and Agriculture Organization](https://www.fao.org/home/en/) and the [World Bank](https://data.worldbank.org/).

The Kaggle page lists the license as **World Bank Dataset Terms of Use**. This is source provenance, not a blanket permission to redistribute a locally recovered copy. The raw CSV therefore remains excluded by `.gitignore`. Before redistributing it, confirm the applicable terms for the exact Kaggle version and the underlying FAO/World Bank data.

For a fresh clone, obtain the permitted dataset copy separately and place it at `data/yield_df.csv` before running training.

## Original project material

This repository includes only materials that are created by or explicitly cleared for publication by Abdellah Guemmah. Before publishing any external dataset or model artifact, confirm the exact source, version, and redistribution terms.

## Required checks before GitHub publication

- Confirm the exact Kaggle dataset version and the underlying source terms.
- Confirm whether the trained model artifacts may be redistributed.
- Record your specific contribution and retain original attribution where a source or artifact requires it.
- Re-run evaluation from a clean environment and record the split and random seed.
