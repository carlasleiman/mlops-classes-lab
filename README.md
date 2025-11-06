# MLOps Lab — From Scripts → Classes

This repo follows the lab where you refactor Titanic ML scripts into well-scoped classes with a package under `src/` and simple CLI scripts under `scripts/`.

## Quickstart

```bash
# 1) (Optional) create venv; if you use uv, it will be auto-created
uv sync

# 2) Run the whole pipeline (expects CSVs in data/raw/)
bash run_pipeline.sh

# Or run steps individually
python scripts/preprocess.py --input data/raw/train.csv --output data/interim/train_preprocessed.parquet
python scripts/featurize.py --input data/interim/train_preprocessed.parquet --output data/processed/train_features.parquet
python scripts/train.py --train data/processed/train_features.parquet --target Survived --model-out models/logreg.joblib --model logreg
python scripts/evaluate.py --val data/processed/train_features.parquet --target Survived --model-in models/logreg.joblib
python scripts/predict.py --input data/processed/train_features.parquet --model-in models/logreg.joblib --pred-out artifacts/preds.csv
```

### Expected layout

```
scripts/
  preprocess.py
  featurize.py
  train.py
  evaluate.py
  predict.py
src/
  mlops_pkg/
    __init__.py
    preprocessing/
      __init__.py
      base_preprocessor.py
      preprocessor.py
    features/
      __init__.py
      base_features_computer.py
      features_computer.py
    models/
      __init__.py
      base_model.py
      logistic_regression.py
      random_forest.py
      xgboost_model.py
```

Place your Titanic CSVs under `data/raw/`:
- `data/raw/train.csv`
- (optional) `data/raw/test.csv`

> `xgboost` is optional. If not installed, that backend will be skipped gracefully.

## GitHub flow (suggested)

- Branch per step (PRs into `main`):
  - `feature/classes-preprocess`
  - `feature/classes-featurize`
  - `feature/classes-models`
  - `feature/model-rf`
  - `feature/model-xgb`

Run `questions.txt`—add your answers there.
