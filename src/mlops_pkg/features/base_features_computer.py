from abc import ABC, abstractmethod
import pandas as pd

class BaseFeaturesComputer(ABC):
    """Abstract base class for feature engineering."""
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
