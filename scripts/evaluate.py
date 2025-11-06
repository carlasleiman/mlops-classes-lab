#!/usr/bin/env python
import argparse, pandas as pd, joblib
from sklearn.metrics import accuracy_score, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--target", default="Survived")
    ap.add_argument("--model-in", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.val)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    model = joblib.load(args.model_in)
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)
        auc = roc_auc_score(y, y_proba)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}  AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
