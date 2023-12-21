from abc import ABC, abstractmethod
import pandas as pd

class Interpolator(ABC):
    """Abstract class for interpolators."""

    def __init__(self, df: pd.DataFrame, target_params: dict):
        self.df = df
        self.target_params = target_params

    @abstractmethod
    def get_design(self) -> pd.DataFrame:
        """Interpolate based on the target parameters.

        Returns:
            pd.DataFrame: DataFrame with interpolated design options.
        """
        pass
