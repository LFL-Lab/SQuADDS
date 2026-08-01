"""Permutation-equivariant Bayesian surrogate for capacitance graphs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .schema import CapacitanceGraph, canonical_edge_index
from .targets import components_to_maxwell, maxwell_to_components

MODEL_FORMAT_VERSION = 1


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _solve_spd(
    matrix: NDArray[np.float64],
    right_hand_side: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve an SPD system and return its inverse with adaptive jitter."""
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(np.diag(symmetric)))), 1.0)
    identity = np.eye(len(symmetric), dtype=np.float64)
    for exponent in range(8):
        jitter = scale * 10.0 ** (-12 + exponent)
        try:
            factor = np.linalg.cholesky(symmetric + jitter * identity)
            intermediate = np.linalg.solve(factor, right_hand_side)
            solution = np.linalg.solve(factor.T, intermediate)
            inverse_intermediate = np.linalg.solve(factor, identity)
            inverse = np.linalg.solve(factor.T, inverse_intermediate)
            return solution, 0.5 * (inverse + inverse.T)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Unable to stabilize the positive-definite linear system.")


@dataclass(frozen=True, slots=True)
class GraphFeatureSignature:
    """Raw numeric descriptor widths expected by a fitted feature map."""

    node_dim: int
    edge_dim: int
    global_dim: int
    parameter_dim: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class TopoCapConfig:
    """Numerical settings for a TopoCap foundation surrogate."""

    random_feature_dimensions: int = 96
    ridge_alpha: float = 1.0
    random_seed: int = 17
    minimum_feature_scale: float = 1e-8
    minimum_noise_variance: float = 1e-6
    standardized_clip: float = 8.0
    log_clip_lower: float = -50.0
    log_clip_upper: float = 50.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.random_feature_dimensions, bool)
            or not isinstance(self.random_feature_dimensions, (int, np.integer))
            or self.random_feature_dimensions < 0
        ):
            raise ValueError("random_feature_dimensions must be a non-negative integer.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, (int, np.integer)):
            raise ValueError("random_seed must be an integer.")
        object.__setattr__(self, "random_feature_dimensions", int(self.random_feature_dimensions))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        for name in (
            "ridge_alpha",
            "minimum_feature_scale",
            "minimum_noise_variance",
            "standardized_clip",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if (
            not np.isfinite([self.log_clip_lower, self.log_clip_upper]).all()
            or self.log_clip_lower >= self.log_clip_upper
        ):
            raise ValueError("log clipping bounds must be finite and increasing.")
        object.__setattr__(self, "log_clip_lower", float(self.log_clip_lower))
        object.__setattr__(self, "log_clip_upper", float(self.log_clip_upper))


@dataclass(frozen=True, slots=True)
class LatentCapacitancePrediction:
    """Node/edge features and Gaussian predictions in positive log space."""

    node_features: NDArray[np.float64]
    edge_features: NDArray[np.float64]
    log_shunt_mean: NDArray[np.float64]
    log_shunt_variance: NDArray[np.float64]
    log_mutual_mean: NDArray[np.float64]
    log_mutual_variance: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CapacitancePrediction:
    """A physical capacitance prediction with marginal uncertainty."""

    edge_index: ArrayLike
    net_ids: Sequence[str]
    log_shunt_mean: ArrayLike
    log_shunt_variance: ArrayLike
    log_mutual_mean: ArrayLike
    log_mutual_variance: ArrayLike
    confidence: float = 0.9
    log_clip: tuple[float, float] = (-50.0, 50.0)

    def __post_init__(self) -> None:
        means_and_variances = {}
        for name in (
            "log_shunt_mean",
            "log_shunt_variance",
            "log_mutual_mean",
            "log_mutual_variance",
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite one-dimensional array.")
            if "variance" in name and np.any(array < 0.0):
                raise ValueError(f"{name} cannot contain negative values.")
            array.setflags(write=False)
            means_and_variances[name] = array

        node_count = len(means_and_variances["log_shunt_mean"])
        if node_count < 2:
            raise ValueError("A prediction must contain at least two nodes.")
        if len(means_and_variances["log_shunt_variance"]) != node_count:
            raise ValueError("Shunt means and variances must have matching lengths.")
        edges = np.array(self.edge_index, dtype=np.int64, copy=True)
        expected_edges = canonical_edge_index(node_count)
        if edges.shape != expected_edges.shape or not np.array_equal(edges, expected_edges):
            raise ValueError("edge_index must be canonical for the predicted node count.")
        edge_count = edges.shape[1]
        if (
            len(means_and_variances["log_mutual_mean"]) != edge_count
            or len(means_and_variances["log_mutual_variance"]) != edge_count
        ):
            raise ValueError("Mutual means and variances must match the canonical edge count.")
        net_ids = tuple(self.net_ids)
        if len(net_ids) != node_count:
            raise ValueError("net_ids must contain one bookkeeping identifier per predicted node.")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one.")
        if len(self.log_clip) != 2 or not np.isfinite(self.log_clip).all() or self.log_clip[0] >= self.log_clip[1]:
            raise ValueError("log_clip must contain finite increasing bounds.")

        edges.setflags(write=False)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "net_ids", net_ids)
        for name, array in means_and_variances.items():
            object.__setattr__(self, name, array)

    def _positive_medians(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.log_clip
        return (
            np.exp(np.clip(self.log_shunt_mean, lower, upper)),
            np.exp(np.clip(self.log_mutual_mean, lower, upper)),
        )

    def _positive_means_and_variances(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.log_clip

        def moments(
            mean: NDArray[np.float64], variance: NDArray[np.float64]
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            safe_mean = np.clip(mean, lower, upper)
            safe_variance = np.clip(variance, 0.0, 50.0)
            expected = np.exp(np.clip(safe_mean + 0.5 * safe_variance, lower, upper))
            value_variance = np.expm1(safe_variance) * np.exp(
                np.clip(2.0 * safe_mean + safe_variance, 2.0 * lower, 2.0 * upper)
            )
            return expected, value_variance

        shunt_mean, shunt_variance = moments(self.log_shunt_mean, self.log_shunt_variance)
        mutual_mean, mutual_variance = moments(self.log_mutual_mean, self.log_mutual_variance)
        return shunt_mean, shunt_variance, mutual_mean, mutual_variance

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Physical posterior-median matrix reconstructed from log medians."""
        shunts, mutuals = self._positive_medians()
        return components_to_maxwell(shunts, mutuals, self.edge_index)

    @property
    def mean_matrix(self) -> NDArray[np.float64]:
        """Physical posterior-mean matrix under marginal log-normal heads."""
        shunt_mean, _, mutual_mean, _ = self._positive_means_and_variances()
        return components_to_maxwell(shunt_mean, mutual_mean, self.edge_index)

    @property
    def matrix_variance(self) -> NDArray[np.float64]:
        """Elementwise marginal variance, assuming independent positive parts."""
        _, shunt_variance, _, mutual_variance = self._positive_means_and_variances()
        variance = np.diag(shunt_variance.copy())
        for (first, second), value in zip(self.edge_index.T, mutual_variance):
            variance[first, first] += value
            variance[second, second] += value
            variance[first, second] = value
            variance[second, first] = value
        return variance

    def interval(self, confidence: float | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return elementwise central log-normal interval bounds.

        The bounds are marginal, not a simultaneous matrix credible region.
        Both the point prediction and every coherent positive-component draw
        remain physical by construction.
        """
        level = self.confidence if confidence is None else float(confidence)
        if not 0.0 < level < 1.0:
            raise ValueError("confidence must lie strictly between zero and one.")
        quantile = NormalDist().inv_cdf(0.5 + level / 2.0)
        lower_clip, upper_clip = self.log_clip

        shunt_lower = np.exp(
            np.clip(self.log_shunt_mean - quantile * np.sqrt(self.log_shunt_variance), lower_clip, upper_clip)
        )
        shunt_upper = np.exp(
            np.clip(self.log_shunt_mean + quantile * np.sqrt(self.log_shunt_variance), lower_clip, upper_clip)
        )
        mutual_lower = np.exp(
            np.clip(self.log_mutual_mean - quantile * np.sqrt(self.log_mutual_variance), lower_clip, upper_clip)
        )
        mutual_upper = np.exp(
            np.clip(self.log_mutual_mean + quantile * np.sqrt(self.log_mutual_variance), lower_clip, upper_clip)
        )

        lower = np.diag(shunt_lower.copy())
        upper = np.diag(shunt_upper.copy())
        for index, (first, second) in enumerate(self.edge_index.T):
            lower[first, first] += mutual_lower[index]
            lower[second, second] += mutual_lower[index]
            upper[first, first] += mutual_upper[index]
            upper[second, second] += mutual_upper[index]
            lower[first, second] = lower[second, first] = -mutual_upper[index]
            upper[first, second] = upper[second, first] = -mutual_lower[index]
        return lower, upper

    @property
    def lower_matrix(self) -> NDArray[np.float64]:
        """Lower bound of the configured marginal interval."""
        return self.interval()[0]

    @property
    def upper_matrix(self) -> NDArray[np.float64]:
        """Upper bound of the configured marginal interval."""
        return self.interval()[1]


def _feature_statistics(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    if matrix.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(
        (
            matrix.mean(axis=0),
            matrix.std(axis=0),
            matrix.min(axis=0),
            matrix.max(axis=0),
            matrix.sum(axis=0) / np.sqrt(len(matrix)),
        )
    )


class EquivariantFeatureBuilder:
    """Construct shared node and symmetric-edge rows from numeric descriptors."""

    @staticmethod
    def infer_signature(samples: Sequence[CapacitanceGraph]) -> GraphFeatureSignature:
        """Infer fixed descriptor widths while allowing absent parameter sets."""
        if not samples:
            raise ValueError("At least one graph is required to infer a feature signature.")
        first = samples[0]
        parameter_dims = {sample.parameter_feature_dim for sample in samples if len(sample.parameter_values)}
        if len(parameter_dims) > 1:
            raise ValueError("Non-empty parameter token features must have a consistent width.")
        parameter_dim = next(iter(parameter_dims), first.parameter_feature_dim)
        signature = GraphFeatureSignature(
            node_dim=first.node_feature_dim,
            edge_dim=first.edge_feature_dim,
            global_dim=first.global_feature_dim,
            parameter_dim=parameter_dim,
        )
        for sample in samples:
            EquivariantFeatureBuilder.validate_signature(sample, signature)
        return signature

    @staticmethod
    def validate_signature(sample: CapacitanceGraph, signature: GraphFeatureSignature) -> None:
        """Validate model-facing widths without requiring optional tokens."""
        observed = (sample.node_feature_dim, sample.edge_feature_dim, sample.global_feature_dim)
        expected = (signature.node_dim, signature.edge_dim, signature.global_dim)
        if observed != expected:
            raise ValueError(f"Graph feature dimensions {observed} do not match fitted dimensions {expected}.")
        if len(sample.parameter_values) and sample.parameter_feature_dim != signature.parameter_dim:
            raise ValueError("Parameter token descriptors do not match the fitted feature dimensions.")

    @staticmethod
    def _parameter_summary(
        sample: CapacitanceGraph,
        signature: GraphFeatureSignature,
    ) -> NDArray[np.float64]:
        token_width = 1 + signature.parameter_dim
        if not len(sample.parameter_values):
            return np.zeros(5 * token_width + 1, dtype=np.float64)
        tokens = np.column_stack((sample.parameter_values, sample.parameter_features))
        return np.concatenate((_feature_statistics(tokens), [np.log1p(len(tokens))]))

    @staticmethod
    def build(
        sample: CapacitanceGraph,
        signature: GraphFeatureSignature,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return permutation-equivariant node rows and invariant edge rows."""
        EquivariantFeatureBuilder.validate_signature(sample, signature)
        node_count = sample.node_count
        node_statistics = _feature_statistics(sample.node_features)
        parameter_summary = EquivariantFeatureBuilder._parameter_summary(sample, signature)

        incident_rows: list[list[int]] = [[] for _ in range(node_count)]
        for edge_row, (first, second) in enumerate(sample.edge_index.T):
            incident_rows[int(first)].append(edge_row)
            incident_rows[int(second)].append(edge_row)
        if signature.edge_dim:
            incident_statistics = np.vstack([_feature_statistics(sample.edge_features[rows]) for rows in incident_rows])
            incident_means = np.vstack([sample.edge_features[rows].mean(axis=0) for rows in incident_rows])
        else:
            incident_statistics = np.empty((node_count, 0), dtype=np.float64)
            incident_means = np.empty((node_count, 0), dtype=np.float64)

        shared = np.concatenate((node_statistics, sample.global_features, parameter_summary))
        graph_scalars = np.array([np.log1p(node_count)], dtype=np.float64)
        node_rows = np.column_stack(
            (
                sample.node_features,
                incident_statistics,
                np.broadcast_to(shared, (node_count, len(shared))),
                np.broadcast_to(graph_scalars, (node_count, 1)),
            )
        )

        first_nodes, second_nodes = sample.edge_index
        first_features = sample.node_features[first_nodes]
        second_features = sample.node_features[second_nodes]
        endpoint_features = np.column_stack(
            (
                first_features + second_features,
                np.abs(first_features - second_features),
                first_features * second_features,
            )
        )
        if signature.edge_dim:
            first_incident = incident_means[first_nodes]
            second_incident = incident_means[second_nodes]
            endpoint_incident = np.column_stack(
                (
                    first_incident + second_incident,
                    np.abs(first_incident - second_incident),
                    first_incident * second_incident,
                )
            )
        else:
            endpoint_incident = np.empty((sample.edge_count, 0), dtype=np.float64)
        edge_rows = np.column_stack(
            (
                sample.edge_features,
                endpoint_features,
                endpoint_incident,
                np.broadcast_to(shared, (sample.edge_count, len(shared))),
                np.broadcast_to(graph_scalars, (sample.edge_count, 1)),
            )
        )
        return node_rows, edge_rows


class _RandomFeatureMap:
    def __init__(self, dimensions: int, seed: int, minimum_scale: float, clip: float):
        self.dimensions = int(dimensions)
        self.seed = int(seed)
        self.minimum_scale = float(minimum_scale)
        self.clip = float(clip)
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.weights_: NDArray[np.float64] | None = None
        self.phase_: NDArray[np.float64] | None = None
        self.gamma_: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, values: ArrayLike) -> _RandomFeatureMap:
        matrix = _matrix(values, "values")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < self.minimum_scale] = 1.0
        standardized = np.clip((matrix - self.mean_) / self.scale_, -self.clip, self.clip)

        rng = np.random.default_rng(self.seed)
        sample_size = min(2_000, len(standardized))
        sample = standardized[rng.choice(len(standardized), size=sample_size, replace=False)]
        pair_count = len(sample) // 2
        if pair_count:
            squared_distances = np.sum(
                (sample[:pair_count] - sample[pair_count : 2 * pair_count]) ** 2,
                axis=1,
            )
            positive_distances = squared_distances[squared_distances > 1e-12]
            median_distance = float(np.median(positive_distances)) if len(positive_distances) else 1.0
        else:
            median_distance = 1.0
        self.gamma_ = 1.0 / max(2.0 * median_distance, 1e-12)
        self.weights_ = rng.normal(
            0.0,
            np.sqrt(2.0 * self.gamma_),
            size=(matrix.shape[1], self.dimensions),
        )
        self.phase_ = rng.uniform(0.0, 2.0 * np.pi, size=self.dimensions)
        return self

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None or self.phase_ is None:
            raise RuntimeError("Fit the random feature map before calling transform().")
        matrix = _matrix(values, "values")
        if matrix.shape[1] != len(self.mean_):
            raise ValueError("values do not match the fitted raw feature dimensions.")
        standardized = np.clip((matrix - self.mean_) / self.scale_, -self.clip, self.clip)
        if not self.dimensions:
            return standardized
        nonlinear = np.sqrt(2.0 / self.dimensions) * np.cos(standardized @ self.weights_ + self.phase_)
        return np.column_stack((standardized, nonlinear))


class TopoCapFeatureMap:
    """Fit source-only preprocessing for equivariant node and edge rows."""

    def __init__(self, config: TopoCapConfig):
        self.config = config
        self.signature_: GraphFeatureSignature | None = None
        self.node_map_: _RandomFeatureMap | None = None
        self.edge_map_: _RandomFeatureMap | None = None

    @property
    def is_fitted(self) -> bool:
        return self.signature_ is not None and self.node_map_ is not None and self.edge_map_ is not None

    def fit(self, samples: Sequence[CapacitanceGraph]) -> TopoCapFeatureMap:
        """Fit all normalization and bandwidth statistics from these samples only."""
        graphs = tuple(samples)
        signature = EquivariantFeatureBuilder.infer_signature(graphs)
        raw = [EquivariantFeatureBuilder.build(sample, signature) for sample in graphs]
        node_rows = np.vstack([rows[0] for rows in raw])
        edge_rows = np.vstack([rows[1] for rows in raw])
        node_map = _RandomFeatureMap(
            self.config.random_feature_dimensions,
            self.config.random_seed,
            self.config.minimum_feature_scale,
            self.config.standardized_clip,
        ).fit(node_rows)
        edge_map = _RandomFeatureMap(
            self.config.random_feature_dimensions,
            self.config.random_seed + 1,
            self.config.minimum_feature_scale,
            self.config.standardized_clip,
        ).fit(edge_rows)
        self.signature_ = signature
        self.node_map_ = node_map
        self.edge_map_ = edge_map
        return self

    def transform(self, sample: CapacitanceGraph) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform one graph without changing fitted source statistics."""
        if not self.is_fitted or self.signature_ is None or self.node_map_ is None or self.edge_map_ is None:
            raise RuntimeError("Fit the TopoCap feature map before calling transform().")
        node_rows, edge_rows = EquivariantFeatureBuilder.build(sample, self.signature_)
        return self.node_map_.transform(node_rows), self.edge_map_.transform(edge_rows)


class _BayesianLinearHead:
    def __init__(self, alpha: float, minimum_noise_variance: float):
        self.alpha = float(alpha)
        self.minimum_noise_variance = float(minimum_noise_variance)
        self.weights_: NDArray[np.float64] | None = None
        self.covariance_: NDArray[np.float64] | None = None
        self.noise_variance_: float | None = None
        self.training_count_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.weights_ is not None and self.covariance_ is not None and self.noise_variance_ is not None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> _BayesianLinearHead:
        x = _matrix(features, "features")
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("targets must be a finite vector with one value per feature row.")
        design = np.column_stack((np.ones(len(x)), x))
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[0, 0] = 1e-8
        gram = design.T @ design + self.alpha * penalty
        weights, inverse_gram = _solve_spd(gram, design.T @ y)
        residual = y - design @ weights
        effective_parameters = float(np.trace(inverse_gram @ (design.T @ design)))
        degrees_of_freedom = max(len(y) - effective_parameters, 1.0)
        noise_variance = max(float(residual @ residual) / degrees_of_freedom, self.minimum_noise_variance)
        self.weights_ = weights
        self.covariance_ = noise_variance * inverse_gram
        self.noise_variance_ = noise_variance
        self.training_count_ = len(y)
        return self

    def predict(self, features: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.is_fitted or self.weights_ is None or self.covariance_ is None or self.noise_variance_ is None:
            raise RuntimeError("Fit the Bayesian head before calling predict().")
        x = _matrix(features, "features")
        if x.shape[1] + 1 != len(self.weights_):
            raise ValueError("features do not match the fitted Bayesian head dimensions.")
        design = np.column_stack((np.ones(len(x)), x))
        mean = design @ self.weights_
        epistemic = np.einsum("ij,jk,ik->i", design, self.covariance_, design, optimize=True)
        variance = np.maximum(self.noise_variance_ + epistemic, self.minimum_noise_variance)
        return mean, variance


class TopoCapFoundationModel:
    """Shared Bayesian node/edge surrogate for arbitrary conductor counts."""

    def __init__(self, config: TopoCapConfig | None = None):
        self.config = config or TopoCapConfig()
        self.feature_map_: TopoCapFeatureMap | None = None
        self.shunt_head_: _BayesianLinearHead | None = None
        self.mutual_head_: _BayesianLinearHead | None = None

    @property
    def is_fitted(self) -> bool:
        return bool(
            self.feature_map_ is not None
            and self.feature_map_.is_fitted
            and self.shunt_head_ is not None
            and self.shunt_head_.is_fitted
            and self.mutual_head_ is not None
            and self.mutual_head_.is_fitted
        )

    def fit(self, samples: Sequence[CapacitanceGraph]) -> TopoCapFoundationModel:
        """Fit source preprocessing and physical log-target heads."""
        graphs = tuple(samples)
        if not graphs:
            raise ValueError("At least one training graph is required.")
        if any(not sample.has_target for sample in graphs):
            raise ValueError("Every training graph must have a capacitance_matrix target.")

        feature_map = TopoCapFeatureMap(self.config).fit(graphs)
        transformed = [feature_map.transform(sample) for sample in graphs]
        node_features = np.vstack([features[0] for features in transformed])
        edge_features = np.vstack([features[1] for features in transformed])
        components = [maxwell_to_components(sample.capacitance_matrix) for sample in graphs]
        log_shunts = np.concatenate([target.log_shunts for target in components])
        log_mutuals = np.concatenate([target.log_mutuals for target in components])

        shunt_head = _BayesianLinearHead(
            self.config.ridge_alpha,
            self.config.minimum_noise_variance,
        ).fit(node_features, log_shunts)
        mutual_head = _BayesianLinearHead(
            self.config.ridge_alpha,
            self.config.minimum_noise_variance,
        ).fit(edge_features, log_mutuals)
        self.feature_map_ = feature_map
        self.shunt_head_ = shunt_head
        self.mutual_head_ = mutual_head
        return self

    def transform_features(
        self,
        sample: CapacitanceGraph,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Expose fitted shared features for residual adaptation and analysis."""
        if not self.is_fitted or self.feature_map_ is None:
            raise RuntimeError("Fit the foundation model before transforming features.")
        return self.feature_map_.transform(sample)

    def predict_latent(self, sample: CapacitanceGraph) -> LatentCapacitancePrediction:
        """Predict Gaussian node shunts and edge mutuals in natural-log space."""
        if not self.is_fitted or self.shunt_head_ is None or self.mutual_head_ is None:
            raise RuntimeError("Fit the foundation model before calling predict_latent().")
        node_features, edge_features = self.transform_features(sample)
        shunt_mean, shunt_variance = self.shunt_head_.predict(node_features)
        mutual_mean, mutual_variance = self.mutual_head_.predict(edge_features)
        return LatentCapacitancePrediction(
            node_features=node_features,
            edge_features=edge_features,
            log_shunt_mean=shunt_mean,
            log_shunt_variance=shunt_variance,
            log_mutual_mean=mutual_mean,
            log_mutual_variance=mutual_variance,
        )

    def predict(self, sample: CapacitanceGraph, confidence: float = 0.9) -> CapacitancePrediction:
        """Predict a physical signed Maxwell matrix and marginal intervals."""
        latent = self.predict_latent(sample)
        return CapacitancePrediction(
            edge_index=sample.edge_index,
            net_ids=sample.net_ids,
            log_shunt_mean=latent.log_shunt_mean,
            log_shunt_variance=latent.log_shunt_variance,
            log_mutual_mean=latent.log_mutual_mean,
            log_mutual_variance=latent.log_mutual_variance,
            confidence=confidence,
            log_clip=(self.config.log_clip_lower, self.config.log_clip_upper),
        )

    def predict_many(
        self,
        samples: Sequence[CapacitanceGraph],
        confidence: float = 0.9,
    ) -> list[CapacitancePrediction]:
        """Predict a sequence of variable-size capacitance graphs."""
        return [self.predict(sample, confidence=confidence) for sample in samples]

    def save(self, path: str | Path) -> Path:
        """Save a fitted model in a pickle-free compressed NumPy archive."""
        if (
            not self.is_fitted
            or self.feature_map_ is None
            or self.feature_map_.signature_ is None
            or self.feature_map_.node_map_ is None
            or self.feature_map_.edge_map_ is None
            or self.shunt_head_ is None
            or self.mutual_head_ is None
        ):
            raise RuntimeError("Fit the foundation model before saving it.")
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "format_version": np.array(MODEL_FORMAT_VERSION, dtype=np.int64),
            "config_json": np.array(json.dumps(asdict(self.config), sort_keys=True)),
            "signature": np.array(list(asdict(self.feature_map_.signature_).values()), dtype=np.int64),
        }
        _store_random_map(state, "node", self.feature_map_.node_map_)
        _store_random_map(state, "edge", self.feature_map_.edge_map_)
        _store_head(state, "shunt", self.shunt_head_)
        _store_head(state, "mutual", self.mutual_head_)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **state)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TopoCapFoundationModel:
        """Load a model produced by :meth:`save` without executable pickle data."""
        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as state:
            version = int(state["format_version"])
            if version != MODEL_FORMAT_VERSION:
                raise ValueError(f"Unsupported TopoCap model format version {version}.")
            config = TopoCapConfig(**json.loads(str(state["config_json"].item())))
            signature_values = np.asarray(state["signature"], dtype=np.int64)
            if signature_values.shape != (4,):
                raise ValueError("Invalid feature signature in the TopoCap archive.")
            signature = GraphFeatureSignature(*map(int, signature_values))
            model = cls(config)
            feature_map = TopoCapFeatureMap(config)
            feature_map.signature_ = signature
            feature_map.node_map_ = _load_random_map(state, "node", config, config.random_seed)
            feature_map.edge_map_ = _load_random_map(state, "edge", config, config.random_seed + 1)
            model.feature_map_ = feature_map
            model.shunt_head_ = _load_head(state, "shunt", config)
            model.mutual_head_ = _load_head(state, "mutual", config)
        if not model.is_fitted:
            raise ValueError("The TopoCap archive is incomplete.")
        return model


def _store_random_map(state: dict[str, Any], prefix: str, feature_map: _RandomFeatureMap) -> None:
    if (
        feature_map.mean_ is None
        or feature_map.scale_ is None
        or feature_map.weights_ is None
        or feature_map.phase_ is None
        or feature_map.gamma_ is None
    ):
        raise RuntimeError("Cannot serialize an unfitted random feature map.")
    state[f"{prefix}_mean"] = feature_map.mean_
    state[f"{prefix}_scale"] = feature_map.scale_
    state[f"{prefix}_weights"] = feature_map.weights_
    state[f"{prefix}_phase"] = feature_map.phase_
    state[f"{prefix}_gamma"] = np.array(feature_map.gamma_, dtype=np.float64)


def _load_random_map(
    state: Any,
    prefix: str,
    config: TopoCapConfig,
    seed: int,
) -> _RandomFeatureMap:
    feature_map = _RandomFeatureMap(
        config.random_feature_dimensions,
        seed,
        config.minimum_feature_scale,
        config.standardized_clip,
    )
    feature_map.mean_ = np.asarray(state[f"{prefix}_mean"], dtype=np.float64)
    feature_map.scale_ = np.asarray(state[f"{prefix}_scale"], dtype=np.float64)
    feature_map.weights_ = np.asarray(state[f"{prefix}_weights"], dtype=np.float64)
    feature_map.phase_ = np.asarray(state[f"{prefix}_phase"], dtype=np.float64)
    feature_map.gamma_ = float(state[f"{prefix}_gamma"])
    if feature_map.weights_.shape != (len(feature_map.mean_), config.random_feature_dimensions):
        raise ValueError(f"Invalid {prefix} random-feature weights in the TopoCap archive.")
    return feature_map


def _store_head(state: dict[str, Any], prefix: str, head: _BayesianLinearHead) -> None:
    if head.weights_ is None or head.covariance_ is None or head.noise_variance_ is None:
        raise RuntimeError("Cannot serialize an unfitted Bayesian head.")
    state[f"{prefix}_weights"] = head.weights_
    state[f"{prefix}_covariance"] = head.covariance_
    state[f"{prefix}_noise_variance"] = np.array(head.noise_variance_, dtype=np.float64)
    state[f"{prefix}_training_count"] = np.array(head.training_count_, dtype=np.int64)


def _load_head(state: Any, prefix: str, config: TopoCapConfig) -> _BayesianLinearHead:
    head = _BayesianLinearHead(config.ridge_alpha, config.minimum_noise_variance)
    head.weights_ = np.asarray(state[f"{prefix}_weights"], dtype=np.float64)
    head.covariance_ = np.asarray(state[f"{prefix}_covariance"], dtype=np.float64)
    head.noise_variance_ = float(state[f"{prefix}_noise_variance"])
    head.training_count_ = int(state[f"{prefix}_training_count"])
    if head.covariance_.shape != (len(head.weights_), len(head.weights_)):
        raise ValueError(f"Invalid {prefix} posterior covariance in the TopoCap archive.")
    return head
