from __future__ import annotations
import pandas as pd
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("xgboost is not installed. Install it or skip this backend.") from e
        self.clf = XGBClassifier(eval_metric="logloss", use_label_encoder=False, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.clf.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.clf.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.clf.predict_proba(X)[:, 1]
