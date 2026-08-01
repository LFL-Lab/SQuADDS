"""Validated graph records for topology-general capacitance learning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def canonical_edge_index(node_count: int) -> NDArray[np.int64]:
    """Return all unordered node pairs in deterministic lexicographic order."""
    if isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer)):
        raise TypeError("node_count must be an integer.")
    if node_count < 2:
        raise ValueError("A capacitance graph must contain at least two conductor nodes.")
    rows, columns = np.triu_indices(int(node_count), k=1)
    edge_index = np.vstack((rows, columns)).astype(np.int64, copy=False)
    edge_index.setflags(write=False)
    return edge_index


def _float_array(values: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _edge_index_array(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E); received {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("edge_index must contain integer node indices.") from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.round(numeric)).all():
            raise ValueError("edge_index must contain integer node indices.")
    array = np.array(raw, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CapacitanceGraph:
    """One arbitrary-size conductor graph and its optional Maxwell target.

    ``net_ids`` and ``parameter_names`` are bookkeeping only. The TopoCap
    feature builder intentionally never encodes either field, preventing
    semantic conductor names or layout-tool vocabularies from becoming family
    identifiers. ``parameter_values`` should contain normalized numeric values;
    optional numeric ``parameter_features`` can describe units or token roles.
    """

    node_features: ArrayLike
    edge_index: ArrayLike
    edge_features: ArrayLike
    global_features: ArrayLike = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    parameter_values: ArrayLike = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    parameter_features: ArrayLike = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    net_ids: Sequence[str] = field(default_factory=tuple)
    parameter_names: Sequence[str] = field(default_factory=tuple)
    capacitance_matrix: ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_features = _float_array(self.node_features, "node_features", 2)
        node_count = node_features.shape[0]
        if node_count < 2:
            raise ValueError("node_features must describe at least two conductor nodes.")

        edge_index = _edge_index_array(self.edge_index)
        expected_edges = canonical_edge_index(node_count)
        if edge_index.shape != expected_edges.shape or not np.array_equal(edge_index, expected_edges):
            raise ValueError(
                "edge_index must contain every unordered node pair exactly once in canonical "
                "lexicographic order; use canonical_edge_index(node_count)."
            )

        raw_edge_features = np.asarray(self.edge_features)
        if raw_edge_features.size == 0 and raw_edge_features.shape == (0, 0):
            raw_edge_features = np.empty((edge_index.shape[1], 0), dtype=np.float64)
        edge_features = _float_array(raw_edge_features, "edge_features", 2)
        if edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_features must contain one row for every canonical edge.")

        global_features = _float_array(self.global_features, "global_features", 1)
        parameter_values = _float_array(self.parameter_values, "parameter_values", 1)
        raw_parameter_features = np.asarray(self.parameter_features)
        if raw_parameter_features.size == 0:
            raw_parameter_features = np.empty((len(parameter_values), 0), dtype=np.float64)
        parameter_features = _float_array(raw_parameter_features, "parameter_features", 2)
        if parameter_features.shape[0] != len(parameter_values):
            raise ValueError("parameter_features must contain one row per parameter value.")

        net_ids = tuple(self.net_ids) or tuple(f"net_{index}" for index in range(node_count))
        if len(net_ids) != node_count or any(not isinstance(net_id, str) or not net_id for net_id in net_ids):
            raise ValueError("net_ids must contain one non-empty string per node.")
        if len(set(net_ids)) != node_count:
            raise ValueError("net_ids must be unique within a graph.")

        parameter_names = tuple(self.parameter_names)
        if parameter_names:
            if len(parameter_names) != len(parameter_values):
                raise ValueError("parameter_names must be empty or contain one name per parameter value.")
            if any(not isinstance(name, str) or not name for name in parameter_names):
                raise ValueError("parameter_names must contain non-empty strings.")
            if len(set(parameter_names)) != len(parameter_names):
                raise ValueError("parameter_names must be unique within a graph.")

        matrix: NDArray[np.float64] | None = None
        if self.capacitance_matrix is not None:
            matrix = _float_array(self.capacitance_matrix, "capacitance_matrix", 2)
            if matrix.shape != (node_count, node_count):
                raise ValueError("capacitance_matrix must have shape (N, N) for the graph's N nodes.")
            scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
            tolerance = 1e-10 * scale
            if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=tolerance):
                raise ValueError("capacitance_matrix must be symmetric.")
            if np.any(np.diag(matrix) <= 0.0):
                raise ValueError("capacitance_matrix diagonal entries must be positive.")
            off_diagonal = matrix.copy()
            np.fill_diagonal(off_diagonal, -1.0)
            if np.any(off_diagonal >= 0.0):
                raise ValueError("capacitance_matrix off-diagonal entries must be strictly negative.")
            mutual = -matrix.copy()
            np.fill_diagonal(mutual, 0.0)
            shunts = np.diag(matrix) - mutual.sum(axis=1)
            if np.any(shunts <= tolerance):
                raise ValueError("capacitance_matrix must have strictly positive residual shunts.")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "node_features", node_features)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_features", edge_features)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "parameter_values", parameter_values)
        object.__setattr__(self, "parameter_features", parameter_features)
        object.__setattr__(self, "net_ids", net_ids)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "capacitance_matrix", matrix)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def node_count(self) -> int:
        """Number of conductor nodes in this graph."""
        return int(self.node_features.shape[0])

    @property
    def edge_count(self) -> int:
        """Number of unordered conductor pairs in this complete graph."""
        return int(self.edge_index.shape[1])

    @property
    def node_feature_dim(self) -> int:
        """Width of each raw node descriptor."""
        return int(self.node_features.shape[1])

    @property
    def edge_feature_dim(self) -> int:
        """Width of each raw edge descriptor."""
        return int(self.edge_features.shape[1])

    @property
    def global_feature_dim(self) -> int:
        """Width of the graph-level numeric context."""
        return int(self.global_features.shape[0])

    @property
    def parameter_feature_dim(self) -> int:
        """Width of each numeric parameter-token descriptor."""
        return int(self.parameter_features.shape[1])

    @property
    def has_target(self) -> bool:
        """Whether a signed Maxwell capacitance target is attached."""
        return self.capacitance_matrix is not None

    @property
    def named_parameter_values(self) -> Mapping[str, float]:
        """Return layout feedback values without exposing names to the model."""
        if not self.parameter_names:
            return MappingProxyType({})
        return MappingProxyType(
            {name: float(value) for name, value in zip(self.parameter_names, self.parameter_values)}
        )

    def reorder_nodes(self, order: Sequence[int]) -> CapacitanceGraph:
        """Return a graph whose new node ``i`` is old node ``order[i]``.

        Edge descriptors are carried to the matching unordered pair. This is a
        data reordering operation, not a semantic relabeling operation.
        """
        permutation = np.asarray(order)
        if permutation.ndim != 1 or len(permutation) != self.node_count:
            raise ValueError("order must be a one-dimensional permutation of all node indices.")
        if not np.issubdtype(permutation.dtype, np.integer):
            try:
                numeric_permutation = np.asarray(permutation, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError("order must contain integer node indices.") from error
            if (
                not np.isfinite(numeric_permutation).all()
                or not np.equal(numeric_permutation, np.round(numeric_permutation)).all()
            ):
                raise ValueError("order must contain integer node indices.")
        permutation = permutation.astype(np.int64, copy=False)
        if not np.array_equal(np.sort(permutation), np.arange(self.node_count)):
            raise ValueError("order must contain every node index exactly once.")

        edge_rows = {(int(first), int(second)): index for index, (first, second) in enumerate(self.edge_index.T)}
        reordered_rows = []
        for first, second in canonical_edge_index(self.node_count).T:
            old_pair = tuple(sorted((int(permutation[first]), int(permutation[second]))))
            reordered_rows.append(edge_rows[old_pair])

        matrix = None
        if self.capacitance_matrix is not None:
            matrix = self.capacitance_matrix[np.ix_(permutation, permutation)]
        return CapacitanceGraph(
            node_features=self.node_features[permutation],
            edge_index=canonical_edge_index(self.node_count),
            edge_features=self.edge_features[reordered_rows],
            global_features=self.global_features,
            parameter_values=self.parameter_values,
            parameter_features=self.parameter_features,
            net_ids=tuple(self.net_ids[index] for index in permutation),
            parameter_names=self.parameter_names,
            capacitance_matrix=matrix,
            metadata=self.metadata,
        )

    def with_target(self, capacitance_matrix: ArrayLike) -> CapacitanceGraph:
        """Return an equivalent graph with a validated Maxwell target attached."""
        return CapacitanceGraph(
            node_features=self.node_features,
            edge_index=self.edge_index,
            edge_features=self.edge_features,
            global_features=self.global_features,
            parameter_values=self.parameter_values,
            parameter_features=self.parameter_features,
            net_ids=self.net_ids,
            parameter_names=self.parameter_names,
            capacitance_matrix=capacitance_matrix,
            metadata=self.metadata,
        )

    def without_target(self) -> CapacitanceGraph:
        """Return an equivalent graph suitable for blind inference."""
        return CapacitanceGraph(
            node_features=self.node_features,
            edge_index=self.edge_index,
            edge_features=self.edge_features,
            global_features=self.global_features,
            parameter_values=self.parameter_values,
            parameter_features=self.parameter_features,
            net_ids=self.net_ids,
            parameter_names=self.parameter_names,
            metadata=self.metadata,
        )


def inverse_permutation(order: Sequence[int]) -> NDArray[np.int64]:
    """Return the inverse of an old-index-in-new-order permutation."""
    permutation = np.asarray(order)
    if permutation.ndim != 1 or not np.issubdtype(permutation.dtype, np.integer):
        raise ValueError("order must be a one-dimensional integer permutation.")
    if not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
        raise ValueError("order must contain every index exactly once.")
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int64)
    inverse.setflags(write=False)
    return inverse
