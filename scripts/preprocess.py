#!/usr/bin/env python
import argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from mlops_pkg.preprocessing.preprocessor import SimpleTitanicPreprocessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = SimpleTitanicPreprocessor().process(df)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, args.output)
    print(f"Preprocessed -> {args.output}")

if __name__ == "__main__":
    main()
