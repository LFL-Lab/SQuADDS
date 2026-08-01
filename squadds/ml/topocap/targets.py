"""Physics-preserving transforms for signed Maxwell capacitance matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import canonical_edge_index


def _positive_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MaxwellComponents:
    """Positive shunt and mutual components of a signed Maxwell matrix."""

    shunts: ArrayLike
    mutuals: ArrayLike
    edge_index: ArrayLike

    def __post_init__(self) -> None:
        shunts = _positive_vector(self.shunts, "shunts")
        mutuals = _positive_vector(self.mutuals, "mutuals")
        expected_edges = canonical_edge_index(len(shunts))
        edge_index = np.array(self.edge_index, dtype=np.int64, copy=True)
        if edge_index.shape != expected_edges.shape or not np.array_equal(edge_index, expected_edges):
            raise ValueError("edge_index must be canonical for the number of shunts.")
        if len(mutuals) != expected_edges.shape[1]:
            raise ValueError("mutuals must contain one value for each unordered node pair.")
        edge_index.setflags(write=False)
        object.__setattr__(self, "shunts", shunts)
        object.__setattr__(self, "mutuals", mutuals)
        object.__setattr__(self, "edge_index", edge_index)

    @property
    def node_count(self) -> int:
        """Number of conductor nodes represented by these components."""
        return int(len(self.shunts))

    @property
    def log_shunts(self) -> NDArray[np.float64]:
        """Natural logarithm of each positive residual shunt."""
        values = np.log(self.shunts)
        values.setflags(write=False)
        return values

    @property
    def log_mutuals(self) -> NDArray[np.float64]:
        """Natural logarithm of each positive pairwise mutual magnitude."""
        values = np.log(self.mutuals)
        values.setflags(write=False)
        return values

    def to_matrix(self) -> NDArray[np.float64]:
        """Reconstruct the signed Maxwell matrix exactly."""
        return components_to_maxwell(self.shunts, self.mutuals, self.edge_index)


@dataclass(frozen=True, slots=True)
class MaxwellDiagnostics:
    """Numerical physicality margins for one signed Maxwell matrix."""

    symmetric: bool
    positive_diagonal: bool
    nonpositive_off_diagonal: bool
    positive_shunts: bool
    positive_semidefinite: bool
    minimum_diagonal: float
    minimum_mutual: float
    minimum_shunt: float
    minimum_eigenvalue: float

    @property
    def is_physical(self) -> bool:
        """Whether all signed-Maxwell physicality checks pass."""
        return bool(
            self.symmetric
            and self.positive_diagonal
            and self.nonpositive_off_diagonal
            and self.positive_shunts
            and self.positive_semidefinite
        )


def maxwell_to_components(matrix: ArrayLike) -> MaxwellComponents:
    """Factor a physical signed Maxwell matrix into positive graph targets.

    For each unordered pair ``(i, j)``, ``m_ij = -C_ij``. The residual
    conductor-to-reference shunt is ``s_i = C_ii - sum_j m_ij``. Strict
    positivity is required because TopoCap learns ``log(m_ij)`` and
    ``log(s_i)``.
    """
    capacitance = np.asarray(matrix, dtype=np.float64)
    if capacitance.ndim != 2 or capacitance.shape[0] != capacitance.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array.")
    if len(capacitance) < 2:
        raise ValueError("matrix must contain at least two conductor nodes.")
    if not np.isfinite(capacitance).all():
        raise ValueError("matrix must contain only finite values.")

    scale = max(float(np.max(np.abs(capacitance))), np.finfo(np.float64).tiny)
    tolerance = 1e-10 * scale
    if not np.allclose(capacitance, capacitance.T, rtol=1e-10, atol=tolerance):
        raise ValueError("matrix must be symmetric.")
    if np.any(np.diag(capacitance) <= 0.0):
        raise ValueError("matrix diagonal entries must be strictly positive.")

    edges = canonical_edge_index(len(capacitance))
    mutuals = -capacitance[edges[0], edges[1]]
    if np.any(mutuals <= 0.0):
        raise ValueError("matrix off-diagonal entries must be strictly negative.")
    incident_mutual = np.zeros(len(capacitance), dtype=np.float64)
    np.add.at(incident_mutual, edges[0], mutuals)
    np.add.at(incident_mutual, edges[1], mutuals)
    shunts = np.diag(capacitance) - incident_mutual
    if np.any(shunts <= tolerance):
        raise ValueError("matrix must have strictly positive residual shunts.")
    return MaxwellComponents(shunts=shunts, mutuals=mutuals, edge_index=edges)


def components_to_maxwell(
    shunts: ArrayLike,
    mutuals: ArrayLike,
    edge_index: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Build a physical signed Maxwell matrix from positive components.

    The construction guarantees symmetry, positive diagonal entries,
    nonpositive off-diagonal entries, strict diagonal dominance by each shunt,
    and positive definiteness. Specifically,

    ``x.T @ C @ x = sum_ij m_ij (x_i - x_j)^2 + sum_i s_i x_i^2``.
    """
    shunt_values = _positive_vector(shunts, "shunts")
    mutual_values = _positive_vector(mutuals, "mutuals")
    edges = canonical_edge_index(len(shunt_values)) if edge_index is None else np.asarray(edge_index, dtype=np.int64)
    expected_edges = canonical_edge_index(len(shunt_values))
    if edges.shape != expected_edges.shape or not np.array_equal(edges, expected_edges):
        raise ValueError("edge_index must be canonical for the number of shunts.")
    if len(mutual_values) != edges.shape[1]:
        raise ValueError("mutuals must contain one value for every unordered node pair.")

    matrix = np.diag(np.asarray(shunt_values).copy())
    for edge, mutual in zip(edges.T, mutual_values):
        first, second = int(edge[0]), int(edge[1])
        matrix[first, first] += mutual
        matrix[second, second] += mutual
        matrix[first, second] = -mutual
        matrix[second, first] = -mutual
    return matrix


def logs_to_maxwell(
    log_shunts: ArrayLike,
    log_mutuals: ArrayLike,
    edge_index: ArrayLike | None = None,
    log_clip: tuple[float, float] = (-50.0, 50.0),
) -> NDArray[np.float64]:
    """Exponentiate learned targets and reconstruct a physical matrix."""
    lower, upper = log_clip
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("log_clip must contain finite increasing bounds.")
    shunt_logs = np.asarray(log_shunts, dtype=np.float64)
    mutual_logs = np.asarray(log_mutuals, dtype=np.float64)
    if not np.isfinite(shunt_logs).all() or not np.isfinite(mutual_logs).all():
        raise ValueError("log targets must contain only finite values.")
    return components_to_maxwell(
        np.exp(np.clip(shunt_logs, lower, upper)),
        np.exp(np.clip(mutual_logs, lower, upper)),
        edge_index,
    )


def maxwell_diagnostics(matrix: ArrayLike) -> MaxwellDiagnostics:
    """Return scale-aware physicality diagnostics without raising."""
    capacitance = np.asarray(matrix, dtype=np.float64)
    if (
        capacitance.ndim != 2
        or capacitance.shape[0] != capacitance.shape[1]
        or len(capacitance) < 2
        or not np.isfinite(capacitance).all()
    ):
        return MaxwellDiagnostics(False, False, False, False, False, np.nan, np.nan, np.nan, np.nan)

    scale = max(float(np.max(np.abs(capacitance))), np.finfo(np.float64).tiny)
    tolerance = 1e-10 * scale
    symmetric = bool(np.allclose(capacitance, capacitance.T, rtol=1e-10, atol=tolerance))
    diagonal = np.diag(capacitance)
    edges = canonical_edge_index(len(capacitance))
    mutuals = -capacitance[edges[0], edges[1]]
    incident_mutual = np.zeros(len(capacitance), dtype=np.float64)
    np.add.at(incident_mutual, edges[0], mutuals)
    np.add.at(incident_mutual, edges[1], mutuals)
    shunts = diagonal - incident_mutual
    eigenvalues = np.linalg.eigvalsh(0.5 * (capacitance + capacitance.T))
    return MaxwellDiagnostics(
        symmetric=symmetric,
        positive_diagonal=bool(np.all(diagonal > 0.0)),
        nonpositive_off_diagonal=bool(np.all(mutuals >= -tolerance)),
        positive_shunts=bool(np.all(shunts > tolerance)),
        positive_semidefinite=bool(eigenvalues[0] >= -tolerance),
        minimum_diagonal=float(np.min(diagonal)),
        minimum_mutual=float(np.min(mutuals)),
        minimum_shunt=float(np.min(shunts)),
        minimum_eigenvalue=float(eigenvalues[0]),
    )
