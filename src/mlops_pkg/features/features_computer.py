from __future__ import annotations
import pandas as pd
from .base_features_computer import BaseFeaturesComputer

class TitanicFeatures(BaseFeaturesComputer):
    """Very small feature set suitable for demo."""
    CATS = ["Sex", "Embarked"]
    NUMS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure columns exist
        for c in self.NUMS + self.CATS:
            if c not in df:
                df[c] = 0 if c in self.NUMS else "Unknown"

        # One-hot encode categoricals
        df_cat = pd.get_dummies(df[self.CATS].astype("category"), drop_first=True)
        # Keep numerics
        df_num = df[self.NUMS].astype("float32")
        out = pd.concat([df_num, df_cat], axis=1)
        # Pass-through target if present
        if "Survived" in df:
            out["Survived"] = df["Survived"].astype("int32")
        return out
