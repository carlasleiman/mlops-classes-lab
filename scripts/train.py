#!/usr/bin/env python
import argparse, pandas as pd, joblib
from mlops_pkg.models.logistic_regression import LogisticRegressionModel
from mlops_pkg.models.random_forest import RandomForestModel
try:
    from mlops_pkg.models.xgboost_model import XGBoostModel
    HAS_XGB = True
except Exception:
    HAS_XGB = False

MODELS = {
    "logreg": LogisticRegressionModel,
    "rf": RandomForestModel,
}
if HAS_XGB:
    MODELS["xgb"] = XGBoostModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--target", default="Survived")
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--model", choices=list(MODELS.keys()), default="logreg")
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    model = MODELS[args.model]()
    model.fit(X, y)
    joblib.dump(model, args.model_out)
    print(f"Saved model -> {args.model_out}")

if __name__ == "__main__":
    main()
