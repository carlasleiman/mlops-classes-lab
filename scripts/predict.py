#!/usr/bin/env python
import argparse, pandas as pd, joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-in", required=True)
    ap.add_argument("--pred-out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    X = df.drop(columns=[c for c in ["Survived"] if c in df])

    model = joblib.load(args.model_in)
    preds = model.predict(X)
    out = pd.DataFrame({"prediction": preds})
    out.to_csv(args.pred_out, index=False)
    print(f"Predictions -> {args.pred_out}")

if __name__ == "__main__":
    main()
