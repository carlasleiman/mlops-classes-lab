from __future__ import annotations
import pandas as pd
from .base_preprocessor import BasePreprocessor

class SimpleTitanicPreprocessor(BasePreprocessor):
    """Minimal, deterministic preprocessing for Titanic CSV."""
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Standardize column names
        df.columns = [c.strip() for c in df.columns]

        # Basic cleaning
        if "Age" in df:
            df["Age"] = df["Age"].fillna(df["Age"].median())
        if "Fare" in df:
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        for col in ["Embarked", "Cabin", "Ticket"]:
            if col in df:
                df[col] = df[col].fillna("Unknown")

        # Encode 'Sex' column to numeric
        if "Sex" in df:
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

        # Drop columns not useful for training
        drop_cols = [c for c in ["Name", "Ticket", "Cabin", "Embarked"] if c in df]
        return df.drop(columns=drop_cols, errors="ignore")
