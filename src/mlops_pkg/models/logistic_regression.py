from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.clf = LogisticRegression(max_iter=1000, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self.clf.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.clf.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)[:, 1]
        # Fallback for models without predict_proba
        return self.clf.decision_function(X)
