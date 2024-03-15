from abc import ABC, abstractmethod
from squadds import Analyzer
import pandas as pd

class Interpolator(ABC):
    """Abstract class for interpolators."""

    def __init__(self, analyzer: Analyzer, target_params: dict):
        self.analyzer = analyzer  # Correct the typo here
        self.df = self.analyzer.df  # And here
        self.target_params = target_params

    @abstractmethod
    def get_design(self) -> pd.DataFrame:
        """Interpolate based on the target parameters.
        Returns:
            pd.DataFrame: DataFrame with interpolated design options.
        """
        pass
