#!/usr/bin/env python
import argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from mlops_pkg.features.features_computer import TitanicFeatures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    feat = TitanicFeatures().transform(df)
    pq.write_table(pa.Table.from_pandas(feat), args.output)
    print(f"Features -> {args.output}")

if __name__ == "__main__":
    main()
