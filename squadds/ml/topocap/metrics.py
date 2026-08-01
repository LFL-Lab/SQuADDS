"""Evaluation metrics for variable-size signed Maxwell capacitance matrices.

All capacitance values are assumed to use one common unit, normally fF.  A
valid signed Maxwell matrix is symmetric, has non-positive off-diagonals, and
is positive semidefinite.  Its positive pair capacitances are ``-C[i, j]`` and
its shunt-to-reference capacitances are the row sums.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

NanPolicy = Literal["raise", "omit", "propagate"]


def _validate_nan_policy(nan_policy: NanPolicy) -> None:
    if nan_policy not in {"raise", "omit", "propagate"}:
        raise ValueError(f"Unknown nan_policy: {nan_policy!r}.")


def _as_matrix_sequence(values: Sequence[np.ndarray] | np.ndarray, *, name: str) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim == 2:
        sequence = [values]
    elif isinstance(values, np.ndarray) and values.ndim == 3:
        sequence = [values[index] for index in range(values.shape[0])]
    else:
        sequence = list(values)
    result: list[np.ndarray] = []
    for index, value in enumerate(sequence):
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name}[{index}] must be a square matrix, got {matrix.shape}.")
        if matrix.shape[0] < 1:
            raise ValueError(f"{name}[{index}] must contain at least one node.")
        result.append(matrix)
    return result


def _paired_matrices(
    y_true: Sequence[np.ndarray] | np.ndarray,
    y_pred: Sequence[np.ndarray] | np.ndarray,
    *,
    nan_policy: NanPolicy,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    _validate_nan_policy(nan_policy)
    truth = _as_matrix_sequence(y_true, name="y_true")
    prediction = _as_matrix_sequence(y_pred, name="y_pred")
    if len(truth) != len(prediction):
        raise ValueError("y_true and y_pred contain different sample counts.")
    for index, (expected, actual) in enumerate(zip(truth, prediction)):
        if expected.shape != actual.shape:
            raise ValueError(f"Matrix shape mismatch at sample {index}: {expected.shape} != {actual.shape}.")
        if nan_policy == "raise" and (not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual))):
            raise ValueError(f"Non-finite value at sample {index} with nan_policy='raise'.")
    return truth, prediction


def _finite_pairs(expected: np.ndarray, actual: np.ndarray, nan_policy: NanPolicy) -> tuple[np.ndarray, np.ndarray]:
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if nan_policy == "omit":
        mask = np.isfinite(expected) & np.isfinite(actual)
        return expected[mask], actual[mask]
    return expected, actual


def _mean(values: Iterable[float], nan_policy: NanPolicy) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return math.nan
    if nan_policy == "omit":
        array = array[np.isfinite(array)]
        return float(np.mean(array)) if array.size else math.nan
    return float(np.mean(array))


def _rmse(expected: np.ndarray, actual: np.ndarray, nan_policy: NanPolicy) -> float:
    expected, actual = _finite_pairs(expected, actual, nan_policy)
    if expected.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(actual - expected))))


def _r2(expected: np.ndarray, actual: np.ndarray, nan_policy: NanPolicy, epsilon: float) -> float:
    expected, actual = _finite_pairs(expected, actual, nan_policy)
    if expected.size < 2:
        return math.nan
    centered = expected - np.mean(expected)
    denominator = float(np.sum(np.square(centered)))
    if denominator <= epsilon:
        return math.nan
    return float(1.0 - np.sum(np.square(actual - expected)) / denominator)


def _relative_errors(expected: np.ndarray, actual: np.ndarray, epsilon: float) -> np.ndarray:
    return np.abs(actual - expected) / np.maximum(np.abs(expected), epsilon)


def _safe_log_error(expected: np.ndarray, actual: np.ndarray, epsilon: float, nan_policy: NanPolicy) -> float:
    expected, actual = _finite_pairs(expected, actual, nan_policy)
    if expected.size == 0:
        return math.nan
    # Invalid negative capacitances receive a large, finite penalty instead of
    # disappearing under an omit policy. Physical diagnostics expose the sign.
    expected_log = np.log(np.maximum(expected, epsilon))
    actual_log = np.log(np.maximum(actual, epsilon))
    return float(np.mean(np.abs(actual_log - expected_log)))


def _unique_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.triu_indices(matrix.shape[0])
    return rows, columns, matrix[rows, columns]


def _edge_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.triu_indices(matrix.shape[0], k=1)
    return rows, columns, -matrix[rows, columns]


def _shunt_values(matrix: np.ndarray) -> np.ndarray:
    return np.sum(matrix, axis=1)


@dataclass(frozen=True)
class PhysicalDiagnostics:
    """Physical-validity diagnostics for one predicted Maxwell matrix."""

    node_count: int
    symmetry_max_abs_ff: float
    positive_offdiagonal_count: int
    negative_diagonal_count: int
    diagonal_dominance_min_margin_ff: float
    minimum_eigenvalue_ff: float
    symmetric: bool
    offdiagonal_nonpositive: bool
    diagonal_nonnegative: bool
    diagonally_dominant: bool
    positive_semidefinite: bool
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "symmetry_max_abs_ff": self.symmetry_max_abs_ff,
            "positive_offdiagonal_count": self.positive_offdiagonal_count,
            "negative_diagonal_count": self.negative_diagonal_count,
            "diagonal_dominance_min_margin_ff": self.diagonal_dominance_min_margin_ff,
            "minimum_eigenvalue_ff": self.minimum_eigenvalue_ff,
            "symmetric": self.symmetric,
            "offdiagonal_nonpositive": self.offdiagonal_nonpositive,
            "diagonal_nonnegative": self.diagonal_nonnegative,
            "diagonally_dominant": self.diagonally_dominant,
            "positive_semidefinite": self.positive_semidefinite,
            "valid": self.valid,
        }


def maxwell_physical_diagnostics(
    matrix: np.ndarray,
    *,
    atol: float = 1.0e-9,
    rtol: float = 1.0e-10,
    eigenvalue_atol: float = 1.0e-9,
) -> PhysicalDiagnostics:
    """Evaluate symmetry, signs, dominance, and PSD for one signed matrix."""
    if atol < 0.0 or rtol < 0.0 or eigenvalue_atol < 0.0:
        raise ValueError("Physical diagnostic tolerances must be non-negative.")
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        raise ValueError("matrix must be a non-empty square array.")
    node_count = value.shape[0]
    if not np.all(np.isfinite(value)):
        return PhysicalDiagnostics(
            node_count=node_count,
            symmetry_max_abs_ff=math.nan,
            positive_offdiagonal_count=-1,
            negative_diagonal_count=-1,
            diagonal_dominance_min_margin_ff=math.nan,
            minimum_eigenvalue_ff=math.nan,
            symmetric=False,
            offdiagonal_nonpositive=False,
            diagonal_nonnegative=False,
            diagonally_dominant=False,
            positive_semidefinite=False,
            valid=False,
        )

    scale = max(float(np.max(np.abs(value))), 1.0)
    comparison_atol = atol + rtol * scale
    eigenvalue_limit = eigenvalue_atol + rtol * scale
    symmetry_error = float(np.max(np.abs(value - value.T)))
    offdiag_rows, offdiag_columns = np.triu_indices(node_count, k=1)
    pair_maximum = np.maximum(
        value[offdiag_rows, offdiag_columns],
        value[offdiag_columns, offdiag_rows],
    )
    positive_offdiagonal_count = int(np.count_nonzero(pair_maximum > comparison_atol))
    diagonal = np.diag(value)
    negative_diagonal_count = int(np.count_nonzero(diagonal < -comparison_atol))
    offdiag_abs_sum = np.sum(np.abs(value), axis=1) - np.abs(diagonal)
    dominance_margin = diagonal - offdiag_abs_sum
    minimum_margin = float(np.min(dominance_margin))
    symmetric_matrix = 0.5 * (value + value.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric_matrix)))

    symmetric = symmetry_error <= comparison_atol
    offdiagonal_nonpositive = positive_offdiagonal_count == 0
    diagonal_nonnegative = negative_diagonal_count == 0
    diagonally_dominant = minimum_margin >= -comparison_atol
    positive_semidefinite = minimum_eigenvalue >= -eigenvalue_limit
    return PhysicalDiagnostics(
        node_count=node_count,
        symmetry_max_abs_ff=symmetry_error,
        positive_offdiagonal_count=positive_offdiagonal_count,
        negative_diagonal_count=negative_diagonal_count,
        diagonal_dominance_min_margin_ff=minimum_margin,
        minimum_eigenvalue_ff=minimum_eigenvalue,
        symmetric=symmetric,
        offdiagonal_nonpositive=offdiagonal_nonpositive,
        diagonal_nonnegative=diagonal_nonnegative,
        diagonally_dominant=diagonally_dominant,
        positive_semidefinite=positive_semidefinite,
        valid=(
            symmetric
            and offdiagonal_nonpositive
            and diagonal_nonnegative
            and diagonally_dominant
            and positive_semidefinite
        ),
    )


@dataclass(frozen=True)
class MaxwellMetricReport:
    """Detailed and aggregated metrics for a variable-node matrix dataset."""

    aggregate: Mapping[str, float]
    per_sample: pd.DataFrame
    per_entry: pd.DataFrame
    per_pair: pd.DataFrame
    per_node: pd.DataFrame
    physical: pd.DataFrame
    units: str = "fF"

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": dict(self.aggregate),
            "per_sample": self.per_sample.to_dict(orient="records"),
            "per_entry": self.per_entry.to_dict(orient="records"),
            "per_pair": self.per_pair.to_dict(orient="records"),
            "per_node": self.per_node.to_dict(orient="records"),
            "physical": self.physical.to_dict(orient="records"),
            "units": self.units,
        }


def _summarize_channels(
    raw: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    nan_policy: NanPolicy,
    epsilon: float,
) -> pd.DataFrame:
    """Summarize positive pair or shunt channels by matrix node indices."""
    if raw.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in raw.groupby(grouper, sort=True):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        expected = group["expected_ff"].to_numpy(dtype=np.float64)
        actual = group["predicted_ff"].to_numpy(dtype=np.float64)
        expected_finite, actual_finite = _finite_pairs(expected, actual, nan_policy)
        relative = _relative_errors(expected_finite, actual_finite, epsilon)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_samples": int(expected_finite.size),
                "mae_ff": (
                    float(np.mean(np.abs(actual_finite - expected_finite))) if expected_finite.size else math.nan
                ),
                "rmse_ff": _rmse(expected, actual, nan_policy),
                "r2": _r2(expected, actual, nan_policy, epsilon),
                "log_mae": _safe_log_error(expected, actual, epsilon, nan_policy),
                "within_1pct": float(np.mean(relative <= 0.01)) if relative.size else math.nan,
                "within_5pct": float(np.mean(relative <= 0.05)) if relative.size else math.nan,
                "within_10pct": float(np.mean(relative <= 0.10)) if relative.size else math.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_maxwell_matrices(
    y_true: Sequence[np.ndarray] | np.ndarray,
    y_pred: Sequence[np.ndarray] | np.ndarray,
    *,
    sample_ids: Sequence[Any] | None = None,
    epsilon: float = 1.0e-12,
    nan_policy: NanPolicy = "raise",
    physical_atol: float = 1.0e-9,
) -> MaxwellMetricReport:
    """Evaluate full signed Maxwell matrices without assuming a fixed node count.

    ``macro_*`` values weight every device equally. ``micro_*`` values pool all
    unique matrix entries, so larger matrices contribute more entries.  Full
    Frobenius metrics retain both symmetric off-diagonal entries, matching the
    mathematical matrix norm.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    truth, prediction = _paired_matrices(y_true, y_pred, nan_policy=nan_policy)
    if not truth:
        raise ValueError("At least one matrix pair is required for evaluation.")
    if sample_ids is None:
        ids = list(range(len(truth)))
    else:
        ids = list(sample_ids)
        if len(ids) != len(truth):
            raise ValueError("sample_ids length does not match matrix count.")

    sample_rows: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    full_expected: list[np.ndarray] = []
    full_actual: list[np.ndarray] = []
    unique_expected: list[np.ndarray] = []
    unique_actual: list[np.ndarray] = []
    edge_expected: list[np.ndarray] = []
    edge_actual: list[np.ndarray] = []
    shunt_expected: list[np.ndarray] = []
    shunt_actual: list[np.ndarray] = []

    for sample_position, (sample_id, expected, actual) in enumerate(zip(ids, truth, prediction)):
        finite_expected, finite_actual = _finite_pairs(expected, actual, nan_policy)
        difference = finite_actual - finite_expected
        expected_norm = float(np.linalg.norm(finite_expected)) if finite_expected.size else math.nan
        error_norm = float(np.linalg.norm(difference)) if difference.size else math.nan
        relative_frobenius = error_norm / max(expected_norm, epsilon) if difference.size else math.nan
        sample_rmse = _rmse(expected, actual, nan_policy)
        rms_scale = float(np.sqrt(np.mean(np.square(finite_expected)))) if finite_expected.size else math.nan
        nrmse = sample_rmse / max(rms_scale, epsilon) if np.isfinite(rms_scale) else math.nan

        rows, columns, expected_unique = _unique_values(expected)
        _, _, actual_unique = _unique_values(actual)
        expected_unique_finite, actual_unique_finite = _finite_pairs(expected_unique, actual_unique, nan_policy)
        relative = _relative_errors(expected_unique_finite, actual_unique_finite, epsilon)

        edge_rows, edge_columns, expected_edges = _edge_values(expected)
        _, _, actual_edges = _edge_values(actual)
        expected_shunts = _shunt_values(expected)
        actual_shunts = _shunt_values(actual)
        edge_errors = np.abs(actual_edges - expected_edges)
        shunt_errors = np.abs(actual_shunts - expected_shunts)
        finite_edge_positions = np.flatnonzero(np.isfinite(edge_errors))
        finite_node_positions = np.flatnonzero(np.isfinite(shunt_errors))
        worst_pair_position = (
            int(finite_edge_positions[np.argmax(edge_errors[finite_edge_positions])])
            if finite_edge_positions.size
            else -1
        )
        worst_node_position = (
            int(finite_node_positions[np.argmax(shunt_errors[finite_node_positions])])
            if finite_node_positions.size
            else -1
        )

        sample_rows.append(
            {
                "sample_position": sample_position,
                "sample_id": sample_id,
                "node_count": expected.shape[0],
                "relative_frobenius": relative_frobenius,
                "nrmse": nrmse,
                "mae_ff": (
                    float(np.mean(np.abs(actual_unique_finite - expected_unique_finite)))
                    if expected_unique_finite.size
                    else math.nan
                ),
                "rmse_ff": _rmse(expected_unique, actual_unique, nan_policy),
                "r2": _r2(expected_unique, actual_unique, nan_policy, epsilon),
                "edge_log_mae": _safe_log_error(expected_edges, actual_edges, epsilon, nan_policy),
                "shunt_log_mae": _safe_log_error(expected_shunts, actual_shunts, epsilon, nan_policy),
                "within_1pct": float(np.mean(relative <= 0.01)) if relative.size else math.nan,
                "within_5pct": float(np.mean(relative <= 0.05)) if relative.size else math.nan,
                "within_10pct": float(np.mean(relative <= 0.10)) if relative.size else math.nan,
                "worst_pair_i": int(edge_rows[worst_pair_position]) if worst_pair_position >= 0 else -1,
                "worst_pair_j": int(edge_columns[worst_pair_position]) if worst_pair_position >= 0 else -1,
                "worst_pair_abs_error_ff": (
                    float(edge_errors[worst_pair_position]) if worst_pair_position >= 0 else math.nan
                ),
                "worst_node_i": worst_node_position,
                "worst_node_shunt_abs_error_ff": (
                    float(shunt_errors[worst_node_position]) if worst_node_position >= 0 else math.nan
                ),
            }
        )

        for row, column, expected_value, actual_value in zip(rows, columns, expected_unique, actual_unique):
            entry_rows.append(
                {
                    "sample_position": sample_position,
                    "sample_id": sample_id,
                    "node_count": expected.shape[0],
                    "i": int(row),
                    "j": int(column),
                    "kind": "node" if row == column else "pair",
                    "expected_ff": float(expected_value),
                    "predicted_ff": float(actual_value),
                }
            )
        for row, column, expected_value, actual_value in zip(edge_rows, edge_columns, expected_edges, actual_edges):
            pair_rows.append(
                {
                    "sample_position": sample_position,
                    "sample_id": sample_id,
                    "node_count": expected.shape[0],
                    "i": int(row),
                    "j": int(column),
                    "expected_ff": float(expected_value),
                    "predicted_ff": float(actual_value),
                }
            )
        for node, (expected_value, actual_value) in enumerate(zip(expected_shunts, actual_shunts)):
            node_rows.append(
                {
                    "sample_position": sample_position,
                    "sample_id": sample_id,
                    "node_count": expected.shape[0],
                    "i": node,
                    "expected_ff": float(expected_value),
                    "predicted_ff": float(actual_value),
                }
            )

        diagnostics = maxwell_physical_diagnostics(actual, atol=physical_atol)
        physical_rows.append({"sample_position": sample_position, "sample_id": sample_id, **diagnostics.to_dict()})
        full_expected.append(expected.reshape(-1))
        full_actual.append(actual.reshape(-1))
        unique_expected.append(expected_unique)
        unique_actual.append(actual_unique)
        edge_expected.append(expected_edges)
        edge_actual.append(actual_edges)
        shunt_expected.append(expected_shunts)
        shunt_actual.append(actual_shunts)

    per_sample = pd.DataFrame(sample_rows)
    raw_entries = pd.DataFrame(entry_rows)
    physical = pd.DataFrame(physical_rows)

    per_entry_rows: list[dict[str, Any]] = []
    if not raw_entries.empty:
        for (row, column, kind), group in raw_entries.groupby(["i", "j", "kind"], sort=True):
            expected = group["expected_ff"].to_numpy(dtype=np.float64)
            actual = group["predicted_ff"].to_numpy(dtype=np.float64)
            expected_finite, actual_finite = _finite_pairs(expected, actual, nan_policy)
            relative = _relative_errors(expected_finite, actual_finite, epsilon)
            per_entry_rows.append(
                {
                    "i": int(row),
                    "j": int(column),
                    "kind": kind,
                    "n_samples": int(expected_finite.size),
                    "mae_ff": (
                        float(np.mean(np.abs(actual_finite - expected_finite))) if expected_finite.size else math.nan
                    ),
                    "rmse_ff": _rmse(expected, actual, nan_policy),
                    "r2": _r2(expected, actual, nan_policy, epsilon),
                    "within_1pct": float(np.mean(relative <= 0.01)) if relative.size else math.nan,
                    "within_5pct": float(np.mean(relative <= 0.05)) if relative.size else math.nan,
                    "within_10pct": float(np.mean(relative <= 0.10)) if relative.size else math.nan,
                }
            )
    per_entry = pd.DataFrame(per_entry_rows)
    per_pair = _summarize_channels(
        pd.DataFrame(pair_rows),
        group_columns=("i", "j"),
        nan_policy=nan_policy,
        epsilon=epsilon,
    )
    per_node = _summarize_channels(
        pd.DataFrame(node_rows),
        group_columns=("i",),
        nan_policy=nan_policy,
        epsilon=epsilon,
    )

    pooled_full_expected, pooled_full_actual = _finite_pairs(
        np.concatenate(full_expected) if full_expected else np.asarray([], dtype=float),
        np.concatenate(full_actual) if full_actual else np.asarray([], dtype=float),
        nan_policy,
    )
    pooled_unique_expected, pooled_unique_actual = _finite_pairs(
        np.concatenate(unique_expected) if unique_expected else np.asarray([], dtype=float),
        np.concatenate(unique_actual) if unique_actual else np.asarray([], dtype=float),
        nan_policy,
    )
    pooled_relative = _relative_errors(pooled_unique_expected, pooled_unique_actual, epsilon)
    pooled_full_scale = (
        float(np.sqrt(np.mean(np.square(pooled_full_expected)))) if pooled_full_expected.size else math.nan
    )
    pooled_full_rmse = _rmse(pooled_full_expected, pooled_full_actual, nan_policy)

    macro_columns = (
        "relative_frobenius",
        "nrmse",
        "mae_ff",
        "rmse_ff",
        "r2",
        "edge_log_mae",
        "shunt_log_mae",
        "within_1pct",
        "within_5pct",
        "within_10pct",
        "worst_pair_abs_error_ff",
        "worst_node_shunt_abs_error_ff",
    )
    aggregate: dict[str, float] = {
        f"macro_{column}": _mean(per_sample[column].to_numpy(), nan_policy) for column in macro_columns
    }
    aggregate.update(
        {
            "micro_relative_frobenius": (
                float(np.linalg.norm(pooled_full_actual - pooled_full_expected))
                / max(float(np.linalg.norm(pooled_full_expected)), epsilon)
                if pooled_full_expected.size
                else math.nan
            ),
            "micro_nrmse": (
                pooled_full_rmse / max(pooled_full_scale, epsilon) if np.isfinite(pooled_full_scale) else math.nan
            ),
            "micro_mae_ff": (
                float(np.mean(np.abs(pooled_unique_actual - pooled_unique_expected)))
                if pooled_unique_expected.size
                else math.nan
            ),
            "micro_rmse_ff": _rmse(pooled_unique_expected, pooled_unique_actual, nan_policy),
            "micro_r2": _r2(pooled_unique_expected, pooled_unique_actual, nan_policy, epsilon),
            "micro_edge_log_mae": _safe_log_error(
                np.concatenate(edge_expected) if edge_expected else np.asarray([], dtype=float),
                np.concatenate(edge_actual) if edge_actual else np.asarray([], dtype=float),
                epsilon,
                nan_policy,
            ),
            "micro_shunt_log_mae": _safe_log_error(
                np.concatenate(shunt_expected) if shunt_expected else np.asarray([], dtype=float),
                np.concatenate(shunt_actual) if shunt_actual else np.asarray([], dtype=float),
                epsilon,
                nan_policy,
            ),
            "micro_within_1pct": (float(np.mean(pooled_relative <= 0.01)) if pooled_relative.size else math.nan),
            "micro_within_5pct": (float(np.mean(pooled_relative <= 0.05)) if pooled_relative.size else math.nan),
            "micro_within_10pct": (float(np.mean(pooled_relative <= 0.10)) if pooled_relative.size else math.nan),
            "physical_valid_rate": _mean(physical.get("valid", pd.Series(dtype=float)), nan_policy),
            "physical_symmetry_rate": _mean(physical.get("symmetric", pd.Series(dtype=float)), nan_policy),
            "physical_sign_rate": _mean(physical.get("offdiagonal_nonpositive", pd.Series(dtype=float)), nan_policy),
            "physical_diagonal_dominance_rate": _mean(
                physical.get("diagonally_dominant", pd.Series(dtype=float)), nan_policy
            ),
            "physical_psd_rate": _mean(physical.get("positive_semidefinite", pd.Series(dtype=float)), nan_policy),
            "worst_pair_mae_ff": (float(per_pair["mae_ff"].max()) if not per_pair.empty else math.nan),
            "worst_node_shunt_mae_ff": (float(per_node["mae_ff"].max()) if not per_node.empty else math.nan),
        }
    )
    return MaxwellMetricReport(
        aggregate=aggregate,
        per_sample=per_sample,
        per_entry=per_entry,
        per_pair=per_pair,
        per_node=per_node,
        physical=physical,
    )


def _cluster_tokens(groups: Sequence[Any], expected_shape: tuple[int, ...]) -> np.ndarray:
    """Return deterministic type-aware labels for statistical clusters."""
    group_array = np.asarray(groups, dtype=object)
    if group_array.shape != expected_shape:
        raise ValueError("groups must match the paired observations.")
    tokens: list[str] = []
    for value in group_array:
        if isinstance(value, np.generic):
            value = value.item()
        tokens.append(f"{type(value).__module__}.{type(value).__qualname__}:{value!r}")
    return np.asarray(tokens, dtype=object)


def negative_transfer_rate(
    baseline: Sequence[float],
    transfer: Sequence[float],
    *,
    groups: Sequence[Any] | None = None,
    higher_is_better: bool = False,
    nan_policy: NanPolicy = "omit",
) -> float:
    """Return the fraction of samples or groups on which transfer is worse."""
    _validate_nan_policy(nan_policy)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    transfer_array = np.asarray(transfer, dtype=np.float64)
    if baseline_array.shape != transfer_array.shape or baseline_array.ndim != 1:
        raise ValueError("baseline and transfer must be equally sized one-dimensional arrays.")
    improvement = transfer_array - baseline_array if higher_is_better else baseline_array - transfer_array
    finite = np.isfinite(improvement)
    if nan_policy == "raise" and not np.all(finite):
        raise ValueError("Non-finite paired values with nan_policy='raise'.")
    if nan_policy == "propagate" and not np.all(finite):
        return math.nan
    if groups is None:
        values = improvement[finite] if nan_policy == "omit" else improvement
    else:
        group_tokens = _cluster_tokens(groups, improvement.shape)
        frame = pd.DataFrame(
            {
                "group": group_tokens,
                "improvement": improvement,
            }
        )
        if nan_policy == "omit":
            frame = frame[finite]
        values = frame.groupby("group", sort=True)["improvement"].mean().to_numpy()
    if values.size == 0:
        return math.nan
    return float(np.mean(values < 0.0))


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Cluster-level paired improvement estimate and confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_groups: int
    n_bootstrap: int
    negative_transfer_rate: float
    probability_nonpositive: float
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def paired_group_bootstrap(
    baseline: Sequence[float],
    transfer: Sequence[float],
    groups: Sequence[Any],
    *,
    higher_is_better: bool = False,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
    nan_policy: NanPolicy = "omit",
) -> PairedBootstrapResult:
    """Bootstrap equal-weight group means of paired transfer improvement.

    Positive improvement always means transfer is better: ``transfer -
    baseline`` for scores and ``baseline - transfer`` for losses.  Resampling
    equal-weight group means prevents a large morphology cluster from
    masquerading as many independent replicates.
    """
    _validate_nan_policy(nan_policy)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    baseline_array = np.asarray(baseline, dtype=np.float64)
    transfer_array = np.asarray(transfer, dtype=np.float64)
    if baseline_array.ndim != 1 or baseline_array.shape != transfer_array.shape:
        raise ValueError("baseline and transfer must be equally sized one-dimensional arrays.")
    group_tokens = _cluster_tokens(groups, baseline_array.shape)
    improvement = transfer_array - baseline_array if higher_is_better else baseline_array - transfer_array
    finite = np.isfinite(improvement)
    if nan_policy == "raise" and not np.all(finite):
        raise ValueError("Non-finite paired values with nan_policy='raise'.")
    if nan_policy == "propagate" and not np.all(finite):
        return PairedBootstrapResult(
            estimate=math.nan,
            ci_lower=math.nan,
            ci_upper=math.nan,
            confidence=confidence,
            n_groups=0,
            n_bootstrap=n_bootstrap,
            negative_transfer_rate=math.nan,
            probability_nonpositive=math.nan,
            higher_is_better=higher_is_better,
        )
    frame = pd.DataFrame(
        {
            "group": group_tokens,
            "improvement": improvement,
        }
    )
    if nan_policy == "omit":
        frame = frame[finite]
    group_means = frame.groupby("group", sort=True)["improvement"].mean().to_numpy(dtype=np.float64)
    if group_means.size == 0 or (nan_policy == "propagate" and not np.all(np.isfinite(group_means))):
        return PairedBootstrapResult(
            estimate=math.nan,
            ci_lower=math.nan,
            ci_upper=math.nan,
            confidence=confidence,
            n_groups=int(group_means.size),
            n_bootstrap=n_bootstrap,
            negative_transfer_rate=math.nan,
            probability_nonpositive=math.nan,
            higher_is_better=higher_is_better,
        )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        indices = rng.integers(0, len(group_means), size=len(group_means))
        bootstrap[draw] = float(np.mean(group_means[indices]))
    alpha = 0.5 * (1.0 - confidence)
    return PairedBootstrapResult(
        estimate=float(np.mean(group_means)),
        ci_lower=float(np.quantile(bootstrap, alpha)),
        ci_upper=float(np.quantile(bootstrap, 1.0 - alpha)),
        confidence=confidence,
        n_groups=len(group_means),
        n_bootstrap=n_bootstrap,
        negative_transfer_rate=float(np.mean(group_means < 0.0)),
        probability_nonpositive=float(np.mean(bootstrap <= 0.0)),
        higher_is_better=higher_is_better,
    )


def area_under_learning_curve(
    support_sizes: Sequence[int],
    values: Sequence[float],
    *,
    normalize: bool = True,
    nan_policy: NanPolicy = "omit",
) -> float:
    """Integrate a learning curve over ``log1p(support_size)``."""
    _validate_nan_policy(nan_policy)
    sizes = np.asarray(support_sizes, dtype=np.float64)
    scores = np.asarray(values, dtype=np.float64)
    if sizes.ndim != 1 or sizes.shape != scores.shape:
        raise ValueError("support_sizes and values must be equally sized one-dimensional arrays.")
    if np.any(sizes < 0.0):
        raise ValueError("support_sizes must be non-negative.")
    if nan_policy == "raise" and (not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(scores))):
        raise ValueError("Non-finite learning-curve value with nan_policy='raise'.")
    if nan_policy == "omit":
        mask = np.isfinite(sizes) & np.isfinite(scores)
        sizes, scores = sizes[mask], scores[mask]
    if len(sizes) < 2 or (nan_policy == "propagate" and not np.all(np.isfinite(scores))):
        return math.nan
    order = np.argsort(sizes, kind="stable")
    sizes, scores = sizes[order], scores[order]
    if np.any(np.diff(sizes) == 0.0):
        frame = pd.DataFrame({"size": sizes, "score": scores})
        grouped = frame.groupby("size", sort=True)["score"].mean()
        sizes, scores = grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float)
    if len(sizes) < 2:
        return math.nan
    x = np.log1p(sizes)
    area = float(np.trapz(scores, x=x))
    width = float(x[-1] - x[0])
    return area / width if normalize and width > 0.0 else area


def support_at_target(
    support_sizes: Sequence[int],
    values: Sequence[float],
    target: float,
    *,
    higher_is_better: bool,
    enforce_monotonic: bool = True,
) -> float:
    """Interpolate the first support count at which a target is reached."""
    sizes = np.asarray(support_sizes, dtype=np.float64)
    scores = np.asarray(values, dtype=np.float64)
    if sizes.ndim != 1 or sizes.shape != scores.shape:
        raise ValueError("support_sizes and values must be equally sized one-dimensional arrays.")
    if not np.isfinite(target):
        raise ValueError("target must be finite.")
    mask = np.isfinite(sizes) & np.isfinite(scores)
    sizes, scores = sizes[mask], scores[mask]
    if np.any(sizes < 0.0):
        raise ValueError("support_sizes must be non-negative.")
    if sizes.size == 0:
        return math.nan
    order = np.argsort(sizes, kind="stable")
    sizes, scores = sizes[order], scores[order]
    if np.any(np.diff(sizes) == 0.0):
        frame = pd.DataFrame({"size": sizes, "score": scores})
        grouped = frame.groupby("size", sort=True)["score"].mean()
        sizes, scores = grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float)
    if enforce_monotonic:
        scores = np.maximum.accumulate(scores) if higher_is_better else np.minimum.accumulate(scores)
    reached = scores >= target if higher_is_better else scores <= target
    if not np.any(reached):
        return math.inf
    position = int(np.flatnonzero(reached)[0])
    if position == 0:
        return float(sizes[0])
    previous_score, current_score = scores[position - 1], scores[position]
    if current_score == previous_score:
        return float(sizes[position])
    fraction = (target - previous_score) / (current_score - previous_score)
    log_size = np.log1p(sizes[position - 1]) + fraction * (np.log1p(sizes[position]) - np.log1p(sizes[position - 1]))
    return float(np.expm1(log_size))


def m95_support(
    support_sizes: Sequence[int],
    values: Sequence[float],
    *,
    baseline_value: float,
    asymptotic_value: float | None = None,
    higher_is_better: bool,
) -> float:
    """Return labels needed to achieve 95% of baseline-to-asymptote gain."""
    scores = np.asarray(values, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return math.nan
    if asymptotic_value is None:
        asymptotic_value = float(np.max(finite) if higher_is_better else np.min(finite))
    target = baseline_value + 0.95 * (asymptotic_value - baseline_value)
    return support_at_target(
        support_sizes,
        values,
        target,
        higher_is_better=higher_is_better,
    )


def label_efficiency_ratio(
    target_only_sizes: Sequence[int],
    target_only_values: Sequence[float],
    transfer_sizes: Sequence[int],
    transfer_values: Sequence[float],
    *,
    target: float,
    higher_is_better: bool,
) -> float:
    """Return target-only labels divided by transfer labels at one target."""
    target_only = support_at_target(target_only_sizes, target_only_values, target, higher_is_better=higher_is_better)
    transfer = support_at_target(transfer_sizes, transfer_values, target, higher_is_better=higher_is_better)
    if np.isnan(target_only) or np.isnan(transfer):
        return math.nan
    if np.isinf(target_only) and np.isinf(transfer):
        return math.nan
    if np.isinf(target_only):
        return math.inf
    if np.isinf(transfer):
        return 0.0
    if transfer == 0.0:
        return math.inf if target_only > 0.0 else 1.0
    return float(target_only / transfer)


@dataclass(frozen=True)
class LearningCurveBootstrapResult:
    """Cluster-bootstrap comparison of complete paired learning curves."""

    curve: pd.DataFrame
    baseline_aulc: float
    transfer_aulc: float
    aulc_improvement: float
    aulc_ci_lower: float
    aulc_ci_upper: float
    label_efficiency: float
    label_efficiency_ci_lower: float
    label_efficiency_ci_upper: float
    target: float | None
    negative_transfer_rate: float
    probability_nonpositive_aulc: float
    confidence: float
    n_groups: int
    n_observations: int
    n_bootstrap: int
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in self.__dict__.items() if key != "curve"},
            "curve": self.curve.to_dict(orient="records"),
        }


def _mean_curve(values: np.ndarray, *, nan_policy: NanPolicy) -> np.ndarray:
    if nan_policy != "omit":
        return np.mean(values, axis=0)
    finite = np.isfinite(values)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, values, 0.0), axis=0)
    return np.divide(
        total,
        count,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )


def _nearest_quantile(values: np.ndarray, quantile: float) -> float:
    valid = values[~np.isnan(values)]
    if not valid.size:
        return math.nan
    return float(np.quantile(valid, quantile, method="nearest"))


def paired_learning_curve_bootstrap(
    baseline_values: Sequence[Sequence[float]] | np.ndarray,
    transfer_values: Sequence[Sequence[float]] | np.ndarray,
    support_sizes: Sequence[int],
    groups: Sequence[Any],
    *,
    target: float | None = None,
    higher_is_better: bool = False,
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
    nan_policy: NanPolicy = "omit",
) -> LearningCurveBootstrapResult:
    """Compare paired learning curves by resampling independent clusters.

    Rows are repeated observations, such as seeds or layouts, and columns are
    support levels. Rows with the same ``groups`` value are first averaged so
    each morphology or acquisition block receives equal weight. The bootstrap
    then resamples those group curves, yielding uncertainty for both AULC and
    the target-only/transfer label-efficiency ratio.
    """
    _validate_nan_policy(nan_policy)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    baseline = np.asarray(baseline_values, dtype=np.float64)
    transfer = np.asarray(transfer_values, dtype=np.float64)
    sizes = np.asarray(support_sizes, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape != transfer.shape:
        raise ValueError("baseline_values and transfer_values must be equal two-dimensional arrays.")
    if baseline.shape[0] < 1 or baseline.shape[1] < 2:
        raise ValueError("Learning-curve bootstrap needs observations at two or more support levels.")
    if sizes.ndim != 1 or len(sizes) != baseline.shape[1]:
        raise ValueError("support_sizes must contain one value per learning-curve column.")
    if (
        not np.all(np.isfinite(sizes))
        or np.any(sizes < 0.0)
        or np.any(sizes != np.floor(sizes))
        or np.any(np.diff(sizes) <= 0.0)
    ):
        raise ValueError("support_sizes must be finite, non-negative integers in strictly increasing order.")
    if target is not None and not np.isfinite(target):
        raise ValueError("target must be finite when supplied.")

    group_tokens = _cluster_tokens(groups, (baseline.shape[0],))
    paired_finite = np.isfinite(baseline) & np.isfinite(transfer)
    if nan_policy == "raise" and not np.all(paired_finite):
        raise ValueError("Non-finite paired learning-curve value with nan_policy='raise'.")
    if nan_policy == "omit":
        baseline = np.where(paired_finite, baseline, np.nan)
        transfer = np.where(paired_finite, transfer, np.nan)

    unique_groups = sorted(set(group_tokens.tolist()))
    grouped_baseline = np.vstack(
        [_mean_curve(baseline[group_tokens == group], nan_policy=nan_policy) for group in unique_groups]
    )
    grouped_transfer = np.vstack(
        [_mean_curve(transfer[group_tokens == group], nan_policy=nan_policy) for group in unique_groups]
    )
    mean_baseline = _mean_curve(grouped_baseline, nan_policy=nan_policy)
    mean_transfer = _mean_curve(grouped_transfer, nan_policy=nan_policy)

    baseline_aulc = area_under_learning_curve(sizes, mean_baseline, nan_policy=nan_policy)
    transfer_aulc = area_under_learning_curve(sizes, mean_transfer, nan_policy=nan_policy)
    direction = 1.0 if higher_is_better else -1.0
    aulc_improvement = direction * (transfer_aulc - baseline_aulc)
    curve = pd.DataFrame(
        {
            "support_size": sizes.astype(np.int64),
            "baseline": mean_baseline,
            "transfer": mean_transfer,
            "improvement": direction * (mean_transfer - mean_baseline),
        }
    )

    per_group_aulc_improvement: list[float] = []
    for baseline_curve, transfer_curve in zip(grouped_baseline, grouped_transfer):
        baseline_area = area_under_learning_curve(sizes, baseline_curve, nan_policy=nan_policy)
        transfer_area = area_under_learning_curve(sizes, transfer_curve, nan_policy=nan_policy)
        per_group_aulc_improvement.append(direction * (transfer_area - baseline_area))
    group_improvement = np.asarray(per_group_aulc_improvement, dtype=np.float64)
    finite_group_improvement = group_improvement[np.isfinite(group_improvement)]

    label_efficiency = (
        label_efficiency_ratio(
            sizes,
            mean_baseline,
            sizes,
            mean_transfer,
            target=float(target),
            higher_is_better=higher_is_better,
        )
        if target is not None
        else math.nan
    )
    rng = np.random.default_rng(seed)
    bootstrap_aulc = np.empty(n_bootstrap, dtype=np.float64)
    bootstrap_efficiency = np.full(n_bootstrap, np.nan, dtype=np.float64)
    for draw in range(n_bootstrap):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        baseline_curve = _mean_curve(grouped_baseline[sampled], nan_policy=nan_policy)
        transfer_curve = _mean_curve(grouped_transfer[sampled], nan_policy=nan_policy)
        baseline_area = area_under_learning_curve(sizes, baseline_curve, nan_policy=nan_policy)
        transfer_area = area_under_learning_curve(sizes, transfer_curve, nan_policy=nan_policy)
        bootstrap_aulc[draw] = direction * (transfer_area - baseline_area)
        if target is not None:
            bootstrap_efficiency[draw] = label_efficiency_ratio(
                sizes,
                baseline_curve,
                sizes,
                transfer_curve,
                target=float(target),
                higher_is_better=higher_is_better,
            )

    alpha = 0.5 * (1.0 - confidence)
    finite_bootstrap_aulc = bootstrap_aulc[np.isfinite(bootstrap_aulc)]
    return LearningCurveBootstrapResult(
        curve=curve,
        baseline_aulc=baseline_aulc,
        transfer_aulc=transfer_aulc,
        aulc_improvement=aulc_improvement,
        aulc_ci_lower=_nearest_quantile(bootstrap_aulc, alpha),
        aulc_ci_upper=_nearest_quantile(bootstrap_aulc, 1.0 - alpha),
        label_efficiency=label_efficiency,
        label_efficiency_ci_lower=_nearest_quantile(bootstrap_efficiency, alpha),
        label_efficiency_ci_upper=_nearest_quantile(bootstrap_efficiency, 1.0 - alpha),
        target=target,
        negative_transfer_rate=(
            float(np.mean(finite_group_improvement < 0.0)) if finite_group_improvement.size else math.nan
        ),
        probability_nonpositive_aulc=(
            float(np.mean(finite_bootstrap_aulc <= 0.0)) if finite_bootstrap_aulc.size else math.nan
        ),
        confidence=confidence,
        n_groups=len(unique_groups),
        n_observations=baseline.shape[0],
        n_bootstrap=n_bootstrap,
        higher_is_better=higher_is_better,
    )


@dataclass(frozen=True)
class CalibrationReport:
    """Micro/macro interval calibration for variable-size matrix outputs."""

    micro_coverage: float
    macro_coverage: float
    group_macro_coverage: float
    micro_mean_width_ff: float
    macro_mean_width_ff: float
    group_macro_mean_width_ff: float
    median_width_ff: float
    nominal_coverage: float | None
    coverage_error: float | None
    group_coverage_error: float | None
    n_elements: int
    n_samples: int
    n_groups: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def interval_calibration(
    y_true: Sequence[np.ndarray] | np.ndarray,
    lower: Sequence[np.ndarray] | np.ndarray,
    upper: Sequence[np.ndarray] | np.ndarray,
    *,
    nominal_coverage: float | None = None,
    groups: Sequence[Any] | None = None,
    nan_policy: NanPolicy = "omit",
) -> CalibrationReport:
    """Compute micro, device-macro, and equal-cluster interval calibration."""
    truth, lower_values = _paired_matrices(y_true, lower, nan_policy=nan_policy)
    truth_again, upper_values = _paired_matrices(y_true, upper, nan_policy=nan_policy)
    if len(truth) != len(truth_again):
        raise ValueError("Interval inputs have inconsistent lengths.")
    if nominal_coverage is not None and not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1).")
    if groups is None:
        group_tokens = np.asarray([f"sample:{index}" for index in range(len(truth))], dtype=object)
    else:
        group_tokens = _cluster_tokens(groups, (len(truth),))

    sample_coverage: list[float] = []
    sample_width: list[float] = []
    all_covered: list[np.ndarray] = []
    all_widths: list[np.ndarray] = []
    for expected, lower_matrix, upper_matrix in zip(truth, lower_values, upper_values):
        if np.any(lower_matrix > upper_matrix):
            raise ValueError("An interval lower bound exceeds its upper bound.")
        rows, columns = np.triu_indices(expected.shape[0])
        expected_unique = expected[rows, columns]
        lower_unique = lower_matrix[rows, columns]
        upper_unique = upper_matrix[rows, columns]
        mask = np.isfinite(expected_unique) & np.isfinite(lower_unique) & np.isfinite(upper_unique)
        if nan_policy == "raise" and not np.all(mask):
            raise ValueError("Non-finite interval value with nan_policy='raise'.")
        if nan_policy == "propagate" and not np.all(mask):
            return CalibrationReport(
                micro_coverage=math.nan,
                macro_coverage=math.nan,
                group_macro_coverage=math.nan,
                micro_mean_width_ff=math.nan,
                macro_mean_width_ff=math.nan,
                group_macro_mean_width_ff=math.nan,
                median_width_ff=math.nan,
                nominal_coverage=nominal_coverage,
                coverage_error=None,
                group_coverage_error=None,
                n_elements=sum(matrix.shape[0] * (matrix.shape[0] + 1) // 2 for matrix in truth),
                n_samples=len(truth),
                n_groups=len(set(group_tokens.tolist())),
            )
        if nan_policy == "omit":
            expected_unique = expected_unique[mask]
            lower_unique = lower_unique[mask]
            upper_unique = upper_unique[mask]
        covered = (expected_unique >= lower_unique) & (expected_unique <= upper_unique)
        widths = upper_unique - lower_unique
        sample_coverage.append(float(np.mean(covered)) if covered.size else math.nan)
        sample_width.append(float(np.mean(widths)) if widths.size else math.nan)
        all_covered.append(covered.astype(float))
        all_widths.append(widths)

    covered = np.concatenate(all_covered) if all_covered else np.asarray([], dtype=float)
    widths = np.concatenate(all_widths) if all_widths else np.asarray([], dtype=float)
    micro_coverage = float(np.mean(covered)) if covered.size else math.nan
    sample_summary = pd.DataFrame(
        {
            "group": group_tokens,
            "coverage": sample_coverage,
            "mean_width_ff": sample_width,
        }
    )
    group_summary = sample_summary.groupby("group", sort=True, as_index=False).agg(
        coverage=("coverage", "mean"),
        mean_width_ff=("mean_width_ff", "mean"),
    )
    group_macro_coverage = _mean(group_summary["coverage"], nan_policy)
    return CalibrationReport(
        micro_coverage=micro_coverage,
        macro_coverage=_mean(sample_coverage, nan_policy),
        group_macro_coverage=group_macro_coverage,
        micro_mean_width_ff=float(np.mean(widths)) if widths.size else math.nan,
        macro_mean_width_ff=_mean(sample_width, nan_policy),
        group_macro_mean_width_ff=_mean(group_summary["mean_width_ff"], nan_policy),
        median_width_ff=float(np.median(widths)) if widths.size else math.nan,
        nominal_coverage=nominal_coverage,
        coverage_error=(
            micro_coverage - nominal_coverage if nominal_coverage is not None and np.isfinite(micro_coverage) else None
        ),
        group_coverage_error=(
            group_macro_coverage - nominal_coverage
            if nominal_coverage is not None and np.isfinite(group_macro_coverage)
            else None
        ),
        n_elements=int(covered.size),
        n_samples=len(truth),
        n_groups=len(group_summary),
    )


@dataclass(frozen=True)
class RiskCoverageReport:
    """Selective-prediction risk as increasingly uncertain cases are rejected."""

    curve: pd.DataFrame
    aurc: float
    oracle_aurc: float
    excess_aurc: float
    n_items: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "curve": self.curve.to_dict(orient="records"),
            "aurc": self.aurc,
            "oracle_aurc": self.oracle_aurc,
            "excess_aurc": self.excess_aurc,
            "n_items": self.n_items,
        }


def _coverage_area(risk: np.ndarray) -> float:
    if not len(risk):
        return math.nan
    if len(risk) == 1:
        return float(risk[0])
    coverage = np.arange(1, len(risk) + 1, dtype=np.float64) / len(risk)
    x = np.concatenate(([0.0], coverage))
    y = np.concatenate(([risk[0]], risk))
    return float(np.trapz(y, x=x))


def risk_coverage_curve(
    errors: Sequence[float],
    uncertainty: Sequence[float],
    *,
    groups: Sequence[Any] | None = None,
    nan_policy: NanPolicy = "omit",
) -> RiskCoverageReport:
    """Rank predictions by uncertainty and compute retained-set mean error."""
    _validate_nan_policy(nan_policy)
    error_array = np.asarray(errors, dtype=np.float64)
    uncertainty_array = np.asarray(uncertainty, dtype=np.float64)
    if error_array.ndim != 1 or error_array.shape != uncertainty_array.shape:
        raise ValueError("errors and uncertainty must be equally sized one-dimensional arrays.")
    frame = pd.DataFrame({"error": error_array, "uncertainty": uncertainty_array})
    finite = np.isfinite(error_array) & np.isfinite(uncertainty_array)
    if np.any(error_array[finite] < 0.0):
        raise ValueError("errors must be non-negative.")
    if np.any(uncertainty_array[finite] < 0.0):
        raise ValueError("uncertainty must be non-negative.")
    if nan_policy == "raise" and not np.all(finite):
        raise ValueError("Non-finite risk/uncertainty with nan_policy='raise'.")
    if nan_policy == "propagate" and not np.all(finite):
        return RiskCoverageReport(
            curve=pd.DataFrame(columns=["coverage", "risk", "uncertainty_threshold", "n_retained", "oracle_risk"]),
            aurc=math.nan,
            oracle_aurc=math.nan,
            excess_aurc=math.nan,
            n_items=0,
        )
    group_tokens: np.ndarray | None = None
    if groups is not None:
        group_tokens = _cluster_tokens(groups, error_array.shape)
    if nan_policy == "omit":
        frame = frame[finite].copy()
        if group_tokens is not None:
            group_tokens = group_tokens[finite]
    if group_tokens is not None:
        frame["group"] = group_tokens
        frame = frame.groupby("group", sort=True, as_index=False).agg(
            error=("error", "mean"), uncertainty=("uncertainty", "mean")
        )
    if frame.empty:
        return RiskCoverageReport(
            curve=pd.DataFrame(columns=["coverage", "risk", "uncertainty_threshold", "n_retained", "oracle_risk"]),
            aurc=math.nan,
            oracle_aurc=math.nan,
            excess_aurc=math.nan,
            n_items=0,
        )
    frame = frame.sort_values("uncertainty", kind="stable").reset_index(drop=True)
    count = len(frame)
    curve = pd.DataFrame(
        {
            "coverage": np.arange(1, count + 1, dtype=float) / count,
            "risk": np.cumsum(frame["error"].to_numpy(dtype=float)) / np.arange(1, count + 1),
            "uncertainty_threshold": frame["uncertainty"].to_numpy(dtype=float),
            "n_retained": np.arange(1, count + 1, dtype=int),
        }
    )
    oracle_errors = np.sort(frame["error"].to_numpy(dtype=float), kind="stable")
    oracle_risk = np.cumsum(oracle_errors) / np.arange(1, count + 1)
    curve["oracle_risk"] = oracle_risk
    aurc = _coverage_area(curve["risk"].to_numpy(dtype=float))
    oracle_aurc = _coverage_area(oracle_risk)
    return RiskCoverageReport(
        curve=curve,
        aurc=aurc,
        oracle_aurc=oracle_aurc,
        excess_aurc=aurc - oracle_aurc,
        n_items=count,
    )
