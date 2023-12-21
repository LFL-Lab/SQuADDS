from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from numpy import linalg as LA
import logging
logging.basicConfig(level=logging.INFO)

class MetricStrategy(ABC):
    """Abstract class for metric strategies."""

    @abstractmethod
    def calculate(self, target_params: dict, row: pd.Series) -> float:
        """Calculate the distance metric between target parameters and a DataFrame row.

        Args:
            target_params (dict): Dictionary of target parameters.
            row (pd.Series): A row from a DataFrame.

        Returns:
            float: Calculated distance.
        """
        raise NotImplementedError("This method should be overridden by subclass")

class EuclideanMetric(MetricStrategy):
    """Implements the specific Euclidean metric strategy as per your definition."""

    def calculate(self, target_params, df_row):
        """Calculate the custom Euclidean distance between target_params and df_row.

        The Euclidean distance is calculated as: sqrt(sum_i (x_i - x_{target})^2 / x_{target}),
        where x_i are the values in df_row and x_{target} are the target parameters.

        Parameters:
            target_params (dict): The target parameters as a dictionary.
            df_row (pd.Series): A single row from a DataFrame representing a set of parameters.

        Returns:
            float: The custom Euclidean distance.
        """
        distance = 0.0
        for column, target_value in target_params.items():
            if isinstance(target_value, (int, float)):  # Only numerical columns
                distance += ((df_row[column] - target_value)**2 / target_value**2)
        return np.sqrt(distance)

class ManhattanMetric(MetricStrategy):
    """Implements the Manhattan metric strategy."""

    def calculate(self, target_params, df_row):
        """Calculate the Manhattan distance between target_params and df_row.

        Parameters:
            target_params (dict): The target parameters as a dictionary.
            df_row (pd.Series): A single row from a DataFrame representing a set of parameters.

        Returns:
            float: The Manhattan distance.
        """
        target_vector = np.array([target_params[key] for key in target_params])
        row_vector = np.array([df_row[key] for key in target_params])
        return LA.norm(target_vector - row_vector, ord=1)


class ChebyshevMetric(MetricStrategy):
    """Implements the Chebyshev metric strategy."""

    def calculate(self, target_params, df_row):
        """Calculate the Chebyshev distance between target_params and df_row.

        Parameters:
            target_params (dict): The target parameters as a dictionary.
            df_row (pd.Series): A single row from a DataFrame representing a set of parameters.

        Returns:
            float: The Chebyshev distance.
        """
        target_vector = np.array([target_params[key] for key in target_params])
        row_vector = np.array([df_row[key] for key in target_params])
        return LA.norm(target_vector - row_vector, ord=np.inf)


class WeightedEuclideanMetric(MetricStrategy):
    """Concrete class for weighted Euclidean metric."""

    def __init__(self, weights: dict):
        """Initialize the weights.

        Args:
            weights (dict): Dictionary of weights for each parameter.
        """
        self.weights = weights

    def calculate(self, target_params: dict, row: pd.Series) -> float:
        """Calculate the weighted Euclidean distance between target parameters and a DataFrame row.

        Args:
            target_params (dict): Dictionary of target parameters.
            row (pd.Series): A row from a DataFrame.

        Returns:
            float: Calculated weighted Euclidean distance.
        """
        if self.weights is None:
            self.weights = {key: 1 for key in target_params.keys()}
            logging.info(f"\033[1mNOTE TO USER:\033[0m No metric weights provided. Using default weights of 1 for all parameters.")
        distance = 0
        for param, target_value in target_params.items():
            if isinstance(target_value, (int, float)):
                simulated_value = row.get(param, 0)
                weight = self.weights.get(param, 1)
                distance += weight * ((target_value - simulated_value) ** 2) / target_value**2
        return distance

class CustomMetric(MetricStrategy):
    """Implements a custom metric strategy using a user-defined function.

    Example Usage:
        To use a custom Manhattan distance metric, define the function as follows:

        def manhattan_distance(target, simulated):
            return sum(abs(target[key] - simulated.get(key, 0)) for key in target)

        Then, instantiate CustomMetric with this function:

        custom_metric = CustomMetric(manhattan_distance)
    """

    def __init__(self, custom_metric_func):
        """Initialize CustomMetric with a custom metric function.

        Parameters:
            custom_metric_func (callable): User-defined custom metric function.
                                          The function should take two dictionaries as arguments and return a float.
        """
        if custom_metric_func is None:
            raise ValueError('Must provide a custom metric function.')
        self.custom_metric_func = custom_metric_func

    def calculate(self, target_params, df_row):
        """Calculate the custom metric between target_params and df_row using the user-defined function.

        Parameters:
            target_params (dict): The target parameters as a dictionary.
            df_row (pd.Series): A single row from a DataFrame representing a set of parameters.

        Returns:
            float: The custom metric calculated using the user-defined function.
        """
        return self.custom_metric_func(target_params, df_row.to_dict())