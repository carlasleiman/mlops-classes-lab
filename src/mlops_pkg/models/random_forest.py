from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        self.clf = RandomForestClassifier(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.clf.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.clf.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.clf.predict_proba(X)[:, 1]
