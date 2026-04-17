"""Pure search helpers used by the Analyzer compatibility facade."""

from __future__ import annotations

import logging

import pandas as pd

from squadds.core.metrics import (
    ChebyshevMetric,
    CustomMetric,
    EuclideanMetric,
    ManhattanMetric,
    WeightedEuclideanMetric,
)

SUPPORTED_METRICS = ["Euclidean", "Manhattan", "Chebyshev", "Weighted Euclidean", "Custom"]


def get_H_param_keys_for_system(selected_system):
    """Return Hamiltonian parameter keys for the legacy supported systems."""
    if selected_system == "qubit":
        return ["qubit_frequency_GHz", "anharmonicity_MHz"]
    if selected_system == "cavity_claw":
        return ["resonator_type", "cavity_frequency_GHz", "kappa_kHz"]
    if selected_system == "coupler":
        return None
    if (selected_system == ["qubit", "cavity_claw"]) or (selected_system == ["cavity_claw", "qubit"]):
        return [
            "qubit_frequency_GHz",
            "anharmonicity_MHz",
            "resonator_type",
            "cavity_frequency_GHz",
            "kappa_kHz",
            "g_MHz",
        ]
    raise ValueError("Invalid system.")


def remove_resonator_type_from_target_params(target_params: dict, selected_resonator_type: str | None, *, missing_ok: bool):
    """Preserve the legacy half-wave behavior by mutating the target-params dict in place."""
    if selected_resonator_type == "half":
        if missing_ok:
            target_params.pop("resonator_type", None)
        else:
            target_params.pop("resonator_type")
    return target_params


def outside_bounds(df: pd.DataFrame, params: dict, display: bool = True) -> bool:
    """
    Check whether requested numeric or categorical parameters fall outside the available library.
    """
    is_outside_bounds = False
    filtered_df = df.copy()

    for param, value in params.items():
        if param not in df.columns:
            raise ValueError(f"{param} is not a column in dataframe: {df}")

        if isinstance(value, (int, float)):
            if value < df[param].min() or value > df[param].max():
                if display:
                    logging.info(
                        f"\033[1mNOTE TO USER:\033[0m the value \033[1m{value} for {param}\033[0m is outside the bounds of our library.\nIf you find a geometry which corresponds to these values, please consider contributing it! 😁🙏\n"
                    )
                is_outside_bounds = True
        elif isinstance(value, str):
            filtered_df = filtered_df[filtered_df[param] == value]
        else:
            raise ValueError(f"Unsupported type {type(value)} for parameter {param}")

    if filtered_df.empty:
        categorical_params = {key: value for key, value in params.items() if isinstance(value, str)}
        if display and categorical_params:
            logging.info(
                f"\033[1mNOTE TO USER:\033[0m There are no geometries with the specified categorical parameters - \033[1m{categorical_params}\033[0m.\nIf you find a geometry which corresponds to these values, please consider contributing it! 😁🙏\n"
            )
        is_outside_bounds = True

    return is_outside_bounds


def resolve_metric_strategy(metric: str, metric_weights=None, custom_metric_func=None):
    """Return the metric strategy instance for a supported metric name."""
    if metric == "Euclidean":
        return EuclideanMetric()
    if metric == "Manhattan":
        return ManhattanMetric()
    if metric == "Chebyshev":
        return ChebyshevMetric()
    if metric == "Weighted Euclidean":
        return WeightedEuclideanMetric(metric_weights)
    if metric == "Custom":
        return CustomMetric(custom_metric_func)
    raise ValueError("Invalid metric.")


def filter_df_by_target_params(df: pd.DataFrame, target_params: dict) -> pd.DataFrame:
    """Filter a dataframe by the categorical values present in target params."""
    filtered_df = df
    for param, value in target_params.items():
        if isinstance(value, str):
            filtered_df = filtered_df[filtered_df[param] == value]
    return filtered_df


def rank_closest_indices(filtered_df: pd.DataFrame, target_params: dict, metric_strategy, num_top: int) -> pd.Index:
    """Return indices for the closest rows according to the configured metric."""
    distances = metric_strategy.calculate_vectorized(target_params, filtered_df)
    return distances.nsmallest(num_top).index
