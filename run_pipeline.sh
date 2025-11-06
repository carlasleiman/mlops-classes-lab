#!/bin/bash
set -e

echo "🚀 Running full Titanic MLOps pipeline..."

# 1. Preprocess
python scripts/preprocess.py --input data/raw/train.csv --output data/interim/train_preprocessed.parquet

# 2. Feature engineering
python scripts/featurize.py --input data/interim/train_preprocessed.parquet --output data/processed/train_features.parquet

# 3. Train model
python scripts/train.py --train data/processed/train_features.parquet --target Survived --model-out data/models/model.joblib --model logreg

# 4. Evaluate
python scripts/evaluate.py --model data/models/model.joblib --data data/processed/train_features.parquet --target Survived

echo "✅ Pipeline completed successfully!"
