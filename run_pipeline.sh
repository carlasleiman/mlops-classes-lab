#!/usr/bin/env bash
set -euo pipefail

RAW_TRAIN=${1:-data/raw/train.csv}
PREP=data/interim/train_preprocessed.parquet
FEAT=data/processed/train_features.parquet
MODEL=models/logreg.joblib
PRED=artifacts/preds.csv

mkdir -p data/interim data/processed models artifacts

python scripts/preprocess.py --input "$RAW_TRAIN" --output "$PREP"
python scripts/featurize.py --input "$PREP" --output "$FEAT"
python scripts/train.py --train "$FEAT" --target Survived --model-out "$MODEL" --model logreg
python scripts/evaluate.py --val "$FEAT" --target Survived --model-in "$MODEL"
python scripts/predict.py --input "$FEAT" --model-in "$MODEL" --pred-out "$PRED"

echo "Done. Predictions at $PRED"
