"""Closed-form Bayesian residual adaptation for a fitted TopoCap model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import (
    CapacitancePrediction,
    LatentCapacitancePrediction,
    TopoCapFoundationModel,
    _matrix,
    _solve_spd,
)
from .schema import CapacitanceGraph
from .targets import maxwell_to_components

ADAPTER_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class EBRAConfig:
    """Numerical settings for episodic Bayesian residual adaptation."""

    prior_precision: float = 25.0
    minimum_residual_variance: float = 1e-6
    observation_variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, float(value))


class _BayesianResidualHead:
    def __init__(self, config: EBRAConfig):
        self.config = config
        self.weights_: NDArray[np.float64] | None = None
        self.covariance_: NDArray[np.float64] | None = None
        self.residual_variance_: float | None = None
        self.support_count_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.weights_ is not None and self.covariance_ is not None and self.residual_variance_ is not None

    def fit(
        self,
        features: ArrayLike,
        residual_targets: ArrayLike,
        foundation_variance: ArrayLike,
    ) -> _BayesianResidualHead:
        x = _matrix(features, "features")
        residual = np.asarray(residual_targets, dtype=np.float64)
        base_variance = np.asarray(foundation_variance, dtype=np.float64)
        if residual.ndim != 1 or len(residual) != len(x) or not np.isfinite(residual).all():
            raise ValueError("residual_targets must be a finite vector with one value per feature row.")
        if (
            base_variance.ndim != 1
            or len(base_variance) != len(x)
            or not np.isfinite(base_variance).all()
            or np.any(base_variance < 0.0)
        ):
            raise ValueError("foundation_variance must be a finite non-negative vector matching the features.")

        design = np.column_stack((np.ones(len(x)), x))
        residual_variance = max(float(np.mean(residual**2)), self.config.minimum_residual_variance)
        weights = np.zeros(design.shape[1], dtype=np.float64)
        covariance = np.eye(design.shape[1], dtype=np.float64) / self.config.prior_precision
        for _ in range(3):
            observation_variance = np.maximum(
                base_variance + residual_variance,
                self.config.observation_variance_floor,
            )
            inverse_variance = 1.0 / observation_variance
            precision = self.config.prior_precision * np.eye(design.shape[1], dtype=np.float64)
            precision += design.T @ (design * inverse_variance[:, None])
            right_hand_side = design.T @ (residual * inverse_variance)
            weights, covariance = _solve_spd(precision, right_hand_side)
            remaining = residual - design @ weights
            residual_variance = max(
                float(np.mean(remaining**2)),
                self.config.minimum_residual_variance,
            )

        self.weights_ = weights
        self.covariance_ = covariance
        self.residual_variance_ = residual_variance
        self.support_count_ = len(residual)
        return self

    def predict(self, features: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.is_fitted or self.weights_ is None or self.covariance_ is None or self.residual_variance_ is None:
            raise RuntimeError("Fit the residual head before calling predict().")
        x = _matrix(features, "features")
        if x.shape[1] + 1 != len(self.weights_):
            raise ValueError("features do not match the fitted residual head dimensions.")
        design = np.column_stack((np.ones(len(x)), x))
        mean = design @ self.weights_
        epistemic = np.einsum("ij,jk,ik->i", design, self.covariance_, design, optimize=True)
        variance = np.maximum(
            epistemic + self.residual_variance_,
            self.config.minimum_residual_variance,
        )
        return mean, variance


def foundation_fingerprint(model: TopoCapFoundationModel) -> str:
    """Hash all fitted numerical source state used by an EBRA adapter."""
    if (
        not model.is_fitted
        or model.feature_map_ is None
        or model.feature_map_.signature_ is None
        or model.feature_map_.node_map_ is None
        or model.feature_map_.edge_map_ is None
        or model.shunt_head_ is None
        or model.mutual_head_ is None
    ):
        raise RuntimeError("Fit the foundation model before fingerprinting it.")
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(model.config), sort_keys=True).encode("ascii"))
    digest.update(json.dumps(asdict(model.feature_map_.signature_), sort_keys=True).encode("ascii"))
    arrays = (
        model.feature_map_.node_map_.mean_,
        model.feature_map_.node_map_.scale_,
        model.feature_map_.node_map_.weights_,
        model.feature_map_.node_map_.phase_,
        model.feature_map_.edge_map_.mean_,
        model.feature_map_.edge_map_.scale_,
        model.feature_map_.edge_map_.weights_,
        model.feature_map_.edge_map_.phase_,
        model.shunt_head_.weights_,
        model.shunt_head_.covariance_,
        model.mutual_head_.weights_,
        model.mutual_head_.covariance_,
    )
    for array in arrays:
        if array is None:
            raise RuntimeError("The foundation model contains incomplete fitted state.")
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update(np.asarray(model.shunt_head_.noise_variance_, dtype=np.float64).tobytes())
    digest.update(np.asarray(model.mutual_head_.noise_variance_, dtype=np.float64).tobytes())
    return digest.hexdigest()


class EBRAAdapter:
    """Few-shot residual posterior attached to an immutable foundation model."""

    def __init__(
        self,
        foundation: TopoCapFoundationModel,
        config: EBRAConfig | None = None,
    ):
        if not foundation.is_fitted:
            raise RuntimeError("Fit the foundation model before constructing an adapter.")
        self.foundation = foundation
        self.config = config or EBRAConfig()
        self.shunt_residual_: _BayesianResidualHead | None = None
        self.mutual_residual_: _BayesianResidualHead | None = None
        self.foundation_fingerprint_: str | None = None
        self.support_graph_count_: int = 0

    @property
    def is_fitted(self) -> bool:
        return bool(
            self.foundation_fingerprint_ is not None
            and self.shunt_residual_ is not None
            and self.shunt_residual_.is_fitted
            and self.mutual_residual_ is not None
            and self.mutual_residual_.is_fitted
        )

    def _validate_foundation(self) -> None:
        if self.foundation_fingerprint_ is None:
            raise RuntimeError("Fit the EBRA adapter before prediction.")
        if foundation_fingerprint(self.foundation) != self.foundation_fingerprint_:
            raise RuntimeError("The foundation model has changed since this EBRA adapter was fitted.")

    def fit(self, support_samples: Sequence[CapacitanceGraph]) -> EBRAAdapter:
        """Fit independent node/edge residual posteriors from a target support set."""
        support = tuple(support_samples)
        if not support:
            raise ValueError("At least one target support graph is required.")
        if any(not sample.has_target for sample in support):
            raise ValueError("Every support graph must have a capacitance_matrix target.")

        fingerprint_before = foundation_fingerprint(self.foundation)
        latent = [self.foundation.predict_latent(sample) for sample in support]
        targets = [maxwell_to_components(sample.capacitance_matrix) for sample in support]
        node_features = np.vstack([prediction.node_features for prediction in latent])
        edge_features = np.vstack([prediction.edge_features for prediction in latent])
        shunt_residuals = np.concatenate(
            [target.log_shunts - prediction.log_shunt_mean for target, prediction in zip(targets, latent)]
        )
        mutual_residuals = np.concatenate(
            [target.log_mutuals - prediction.log_mutual_mean for target, prediction in zip(targets, latent)]
        )
        shunt_variance = np.concatenate([prediction.log_shunt_variance for prediction in latent])
        mutual_variance = np.concatenate([prediction.log_mutual_variance for prediction in latent])

        shunt_head = _BayesianResidualHead(self.config).fit(
            node_features,
            shunt_residuals,
            shunt_variance,
        )
        mutual_head = _BayesianResidualHead(self.config).fit(
            edge_features,
            mutual_residuals,
            mutual_variance,
        )
        fingerprint_after = foundation_fingerprint(self.foundation)
        if fingerprint_after != fingerprint_before:
            raise RuntimeError("Foundation state changed unexpectedly during residual adaptation.")
        self.shunt_residual_ = shunt_head
        self.mutual_residual_ = mutual_head
        self.foundation_fingerprint_ = fingerprint_before
        self.support_graph_count_ = len(support)
        return self

    def predict_latent(self, sample: CapacitanceGraph) -> LatentCapacitancePrediction:
        """Combine frozen source predictions with target residual posteriors."""
        if not self.is_fitted or self.shunt_residual_ is None or self.mutual_residual_ is None:
            raise RuntimeError("Fit the EBRA adapter before calling predict_latent().")
        self._validate_foundation()
        base = self.foundation.predict_latent(sample)
        shunt_correction, shunt_adapter_variance = self.shunt_residual_.predict(base.node_features)
        mutual_correction, mutual_adapter_variance = self.mutual_residual_.predict(base.edge_features)
        return LatentCapacitancePrediction(
            node_features=base.node_features,
            edge_features=base.edge_features,
            log_shunt_mean=base.log_shunt_mean + shunt_correction,
            log_shunt_variance=base.log_shunt_variance + shunt_adapter_variance,
            log_mutual_mean=base.log_mutual_mean + mutual_correction,
            log_mutual_variance=base.log_mutual_variance + mutual_adapter_variance,
        )

    def predict(self, sample: CapacitanceGraph, confidence: float = 0.9) -> CapacitancePrediction:
        """Predict an adapted physical matrix and marginal credible intervals."""
        latent = self.predict_latent(sample)
        return CapacitancePrediction(
            edge_index=sample.edge_index,
            net_ids=sample.net_ids,
            log_shunt_mean=latent.log_shunt_mean,
            log_shunt_variance=latent.log_shunt_variance,
            log_mutual_mean=latent.log_mutual_mean,
            log_mutual_variance=latent.log_mutual_variance,
            confidence=confidence,
            log_clip=(self.foundation.config.log_clip_lower, self.foundation.config.log_clip_upper),
        )

    def predict_many(
        self,
        samples: Sequence[CapacitanceGraph],
        confidence: float = 0.9,
    ) -> list[CapacitancePrediction]:
        """Predict a sequence of variable-size target graphs."""
        return [self.predict(sample, confidence=confidence) for sample in samples]

    def save(self, path: str | Path) -> Path:
        """Save residual state; the matching foundation checkpoint stays separate."""
        if (
            not self.is_fitted
            or self.shunt_residual_ is None
            or self.mutual_residual_ is None
            or self.foundation_fingerprint_ is None
        ):
            raise RuntimeError("Fit the EBRA adapter before saving it.")
        self._validate_foundation()
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "format_version": np.array(ADAPTER_FORMAT_VERSION, dtype=np.int64),
            "config_json": np.array(json.dumps(asdict(self.config), sort_keys=True)),
            "foundation_fingerprint": np.array(self.foundation_fingerprint_),
            "support_graph_count": np.array(self.support_graph_count_, dtype=np.int64),
        }
        _store_residual_head(state, "shunt", self.shunt_residual_)
        _store_residual_head(state, "mutual", self.mutual_residual_)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **state)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        foundation: TopoCapFoundationModel,
    ) -> EBRAAdapter:
        """Load residual state and require its exact fitted foundation model."""
        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as state:
            version = int(state["format_version"])
            if version != ADAPTER_FORMAT_VERSION:
                raise ValueError(f"Unsupported EBRA adapter format version {version}.")
            adapter = cls(foundation, EBRAConfig(**json.loads(str(state["config_json"].item()))))
            adapter.foundation_fingerprint_ = str(state["foundation_fingerprint"].item())
            adapter.support_graph_count_ = int(state["support_graph_count"])
            adapter.shunt_residual_ = _load_residual_head(state, "shunt", adapter.config)
            adapter.mutual_residual_ = _load_residual_head(state, "mutual", adapter.config)
        adapter._validate_foundation()
        return adapter


def adapt_foundation(
    foundation: TopoCapFoundationModel,
    support_samples: Sequence[CapacitanceGraph],
    config: EBRAConfig | None = None,
) -> EBRAAdapter:
    """Construct and fit an EBRA adapter without mutating ``foundation``."""
    return EBRAAdapter(foundation, config=config).fit(support_samples)


def _store_residual_head(state: dict[str, Any], prefix: str, head: _BayesianResidualHead) -> None:
    if head.weights_ is None or head.covariance_ is None or head.residual_variance_ is None:
        raise RuntimeError("Cannot serialize an unfitted residual head.")
    state[f"{prefix}_weights"] = head.weights_
    state[f"{prefix}_covariance"] = head.covariance_
    state[f"{prefix}_residual_variance"] = np.array(head.residual_variance_, dtype=np.float64)
    state[f"{prefix}_support_count"] = np.array(head.support_count_, dtype=np.int64)


def _load_residual_head(state: Any, prefix: str, config: EBRAConfig) -> _BayesianResidualHead:
    head = _BayesianResidualHead(config)
    head.weights_ = np.asarray(state[f"{prefix}_weights"], dtype=np.float64)
    head.covariance_ = np.asarray(state[f"{prefix}_covariance"], dtype=np.float64)
    head.residual_variance_ = float(state[f"{prefix}_residual_variance"])
    head.support_count_ = int(state[f"{prefix}_support_count"])
    if head.covariance_.shape != (len(head.weights_), len(head.weights_)):
        raise ValueError(f"Invalid {prefix} residual covariance in the EBRA archive.")
    return head
