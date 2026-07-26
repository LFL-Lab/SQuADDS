"""Transfer-learning utilities for static v0 layout embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .embeddings import MOMENT_BLOCK_SIZE, PARAMETER_BLOCK_SIZE, SHAPE_SIZE


def _as_2d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def compress_v0_embeddings(embeddings: Any, pooled_shape_size: int = 12) -> np.ndarray:
    """Average-pool v0 shape pixels while retaining parameters and moments."""
    matrix = _as_2d(embeddings, "embeddings")
    shape_offset = PARAMETER_BLOCK_SIZE + MOMENT_BLOCK_SIZE
    expected_dimensions = shape_offset + SHAPE_SIZE * SHAPE_SIZE
    if matrix.shape[1] != expected_dimensions:
        raise ValueError(f"Expected {expected_dimensions} v0 dimensions, received {matrix.shape[1]}.")
    if pooled_shape_size < 1 or SHAPE_SIZE % pooled_shape_size:
        raise ValueError(f"pooled_shape_size must divide {SHAPE_SIZE}.")

    block_size = SHAPE_SIZE // pooled_shape_size
    shape = matrix[:, shape_offset:].reshape(-1, pooled_shape_size, block_size, pooled_shape_size, block_size)
    pooled = shape.mean(axis=(2, 4)).reshape(len(matrix), -1)
    return np.concatenate([matrix[:, :shape_offset], pooled], axis=1)


@dataclass
class V0FeatureProjector:
    """Compress and standardize v0 vectors using source-domain statistics."""

    pooled_shape_size: int = 12
    minimum_scale: float = 1e-6
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, embeddings: Any) -> V0FeatureProjector:
        """Fit source-domain feature normalization."""
        features = compress_v0_embeddings(embeddings, self.pooled_shape_size)
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < self.minimum_scale] = 1.0
        return self

    def transform(self, embeddings: Any) -> np.ndarray:
        """Project v0 vectors into the fitted compact feature space."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Fit the feature projector before calling transform().")
        features = compress_v0_embeddings(embeddings, self.pooled_shape_size)
        return (features - self.mean_) / self.scale_

    def fit_transform(self, embeddings: Any) -> np.ndarray:
        """Fit source-domain statistics and transform the same vectors."""
        return self.fit(embeddings).transform(embeddings)


@dataclass
class SourceFeatureProjector:
    """Standardize arbitrary features using source-domain statistics only."""

    minimum_scale: float = 1e-6
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: Any) -> SourceFeatureProjector:
        """Fit source-domain feature normalization."""
        matrix = _as_2d(features, "features")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < self.minimum_scale] = 1.0
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Standardize features with the fitted source-domain statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Fit the feature projector before calling transform().")
        matrix = _as_2d(features, "features")
        if matrix.shape[1] != len(self.mean_):
            raise ValueError("features do not match the fitted feature dimensions.")
        return (matrix - self.mean_) / self.scale_

    def fit_transform(self, features: Any) -> np.ndarray:
        """Fit source-domain statistics and transform the same features."""
        return self.fit(features).transform(features)


class TransferRidgeRegressor:
    """Multi-output ridge regression with an optional source-model prior."""

    def __init__(self, alpha: float = 10.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)
        self.weights_: np.ndarray | None = None

    def fit(
        self,
        features: Any,
        targets: Any,
        prior: TransferRidgeRegressor | np.ndarray | None = None,
    ) -> TransferRidgeRegressor:
        """Fit a target head, regularizing toward source weights when supplied."""
        x = _as_2d(features, "features")
        y = _as_2d(targets, "targets")
        if len(x) != len(y):
            raise ValueError("features and targets must contain the same number of rows.")

        augmented = np.column_stack([np.ones(len(x)), x])
        if isinstance(prior, TransferRidgeRegressor):
            if prior.weights_ is None:
                raise RuntimeError("Fit the prior model before using it.")
            prior_weights = prior.weights_
        elif prior is None:
            prior_weights = np.zeros((augmented.shape[1], y.shape[1]), dtype=np.float64)
        else:
            prior_weights = np.asarray(prior, dtype=np.float64)
        if prior_weights.shape != (augmented.shape[1], y.shape[1]):
            raise ValueError("prior weights do not match the feature and target dimensions.")

        penalty = np.eye(augmented.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        residual = y - augmented @ prior_weights
        update = np.linalg.solve(
            augmented.T @ augmented + self.alpha * penalty,
            augmented.T @ residual,
        )
        self.weights_ = prior_weights + update
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Predict one or more target quantities."""
        if self.weights_ is None:
            raise RuntimeError("Fit the model before calling predict().")
        x = _as_2d(features, "features")
        if x.shape[1] + 1 != self.weights_.shape[0]:
            raise ValueError("features do not match the fitted model dimensions.")
        return np.column_stack([np.ones(len(x)), x]) @ self.weights_


def target_to_source_similarity(source_embeddings: Any, target_embeddings: Any) -> np.ndarray:
    """Return each target vector's cosine similarity to the normalized source centroid."""
    source = _as_2d(source_embeddings, "source_embeddings")
    target = _as_2d(target_embeddings, "target_embeddings")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target embeddings must have the same dimensions.")

    centroid = source.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm <= 1e-12:
        raise ValueError("The source centroid has zero norm.")
    centroid /= centroid_norm
    target_norms = np.linalg.norm(target, axis=1)
    if np.any(target_norms <= 1e-12):
        raise ValueError("Target embeddings must have non-zero norm.")
    return (target @ centroid) / target_norms


def regression_scores(
    expected: Any,
    predicted: Any,
    target_names: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate per-target and macro regression figures of merit."""
    y_true = _as_2d(expected, "expected")
    y_pred = _as_2d(predicted, "predicted")
    if y_true.shape != y_pred.shape:
        raise ValueError("expected and predicted values must have the same shape.")
    names = target_names or [f"target_{index}" for index in range(y_true.shape[1])]
    if len(names) != y_true.shape[1]:
        raise ValueError("target_names must match the number of target columns.")

    records = []
    for index, name in enumerate(names):
        residual = y_true[:, index] - y_pred[:, index]
        denominator = np.sum((y_true[:, index] - y_true[:, index].mean()) ** 2)
        relative_error = np.abs(residual) / np.maximum(np.abs(y_true[:, index]), 1e-12)
        records.append(
            {
                "target": name,
                "r2": 1.0 - float(np.sum(residual**2)) / max(float(denominator), 1e-12),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mape_percent": 100.0 * float(np.mean(relative_error)),
                "within_5_percent": 100.0 * float(np.mean(relative_error <= 0.05)),
            }
        )
    frame = pd.DataFrame(records)
    macro = {"target": "macro", **frame.drop(columns="target").mean().to_dict()}
    return pd.concat([frame, pd.DataFrame([macro])], ignore_index=True)


def _evaluate_models(
    source_model: TransferRidgeRegressor,
    target_train_features: np.ndarray,
    target_train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    alpha: float,
    target_names: list[str],
) -> dict[str, pd.DataFrame]:
    models = {
        "zero-shot": source_model,
        "target-only": TransferRidgeRegressor(alpha).fit(target_train_features, target_train_targets),
        "transfer": TransferRidgeRegressor(alpha).fit(
            target_train_features,
            target_train_targets,
            prior=source_model,
        ),
    }
    return {
        method: regression_scores(test_targets, model.predict(test_features), target_names)
        for method, model in models.items()
    }


def evaluate_transfer_learning(
    source_features: Any,
    source_targets: Any,
    target_features: Any,
    target_targets: Any,
    sample_sizes: list[int],
    *,
    target_names: list[str] | None = None,
    alpha: float = 10.0,
    repeats: int = 12,
    test_fraction: float = 0.3,
    random_seed: int = 15,
) -> pd.DataFrame:
    """Evaluate zero-shot, target-only, and source-prior transfer learning curves."""
    x_source = _as_2d(source_features, "source_features")
    y_source = _as_2d(source_targets, "source_targets")
    x_target = _as_2d(target_features, "target_features")
    y_target = _as_2d(target_targets, "target_targets")
    if len(x_source) != len(y_source) or len(x_target) != len(y_target):
        raise ValueError("Feature and target row counts must match within each domain.")
    if x_source.shape[1] != x_target.shape[1] or y_source.shape[1] != y_target.shape[1]:
        raise ValueError("Source and target domains must share feature and target dimensions.")
    if repeats < 2:
        raise ValueError("repeats must be at least 2.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    names = target_names or [f"target_{index}" for index in range(y_source.shape[1])]
    source_model = TransferRidgeRegressor(alpha).fit(x_source, y_source)
    split_rng = np.random.default_rng(random_seed)
    order = split_rng.permutation(len(x_target))
    test_count = max(1, int(round(len(x_target) * test_fraction)))
    test_indices = order[:test_count]
    pool_indices = order[test_count:]
    sizes = sorted(set(int(size) for size in sample_sizes))
    if not sizes or sizes[0] < 1 or sizes[-1] > len(pool_indices):
        raise ValueError(f"sample_sizes must be between 1 and the target pool size ({len(pool_indices)}).")

    records = []
    for sample_size in sizes:
        for repeat in range(repeats):
            rng = np.random.default_rng(random_seed + 10_000 * sample_size + repeat)
            train_indices = rng.choice(pool_indices, size=sample_size, replace=False)
            scores = _evaluate_models(
                source_model,
                x_target[train_indices],
                y_target[train_indices],
                x_target[test_indices],
                y_target[test_indices],
                alpha,
                names,
            )
            for method, frame in scores.items():
                for record in frame.to_dict(orient="records"):
                    records.append(
                        {
                            "sample_size": sample_size,
                            "repeat": repeat,
                            "method": method,
                            **record,
                        }
                    )
    return pd.DataFrame(records)


def summarize_learning_curve(
    curves: pd.DataFrame,
    *,
    metric: str = "r2",
    confidence: float = 0.8,
) -> pd.DataFrame:
    """Summarize repeated learning curves with a central confidence interval."""
    if metric not in curves:
        raise ValueError(f"Unknown learning-curve metric: {metric!r}.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    group_columns = [
        column
        for column in (
            "source_domain",
            "target_domain",
            "representation",
            "similarity_band",
            "similarity_mean",
            "method",
            "sample_size",
            "target",
        )
        if column in curves
    ]
    tail = (1.0 - confidence) / 2.0
    grouped = curves.groupby(group_columns, dropna=False)[metric]
    return (
        grouped.agg(
            mean="mean",
            lower=lambda values: values.quantile(tail),
            upper=lambda values: values.quantile(1.0 - tail),
        )
        .reset_index()
        .assign(metric=metric, confidence=confidence)
    )


def required_target_samples(
    curves: pd.DataFrame,
    target_scores: list[float],
    *,
    metric: str = "r2",
    method: str = "transfer",
    target: str = "macro",
    confidence: float = 0.8,
) -> pd.DataFrame:
    """Estimate M as the first sample size whose lower confidence bound reaches D."""
    summary = summarize_learning_curve(curves, metric=metric, confidence=confidence)
    filtered = summary.loc[(summary["method"] == method) & (summary["target"] == target)].copy()
    group_columns = [
        column
        for column in (
            "source_domain",
            "target_domain",
            "representation",
            "similarity_band",
            "similarity_mean",
        )
        if column in filtered
    ]
    groups = [((), filtered)] if not group_columns else filtered.groupby(group_columns, dropna=False)

    records = []
    for group_key, frame in groups:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(group_columns, keys))
        ordered = frame.sort_values("sample_size")
        for score in target_scores:
            reached = ordered.loc[ordered["lower"] >= score]
            records.append(
                {
                    **identity,
                    "metric": metric,
                    "target_score": float(score),
                    "required_samples": int(reached.iloc[0]["sample_size"]) if not reached.empty else pd.NA,
                    "method": method,
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(records)


class V0TransferLearningStudy:
    """Prepare source/target v0 domains and run reproducible transfer studies."""

    def __init__(
        self,
        source_embeddings: Any,
        source_targets: Any,
        target_embeddings: Any,
        target_targets: Any,
        *,
        target_names: list[str] | None = None,
        pooled_shape_size: int = 12,
        alpha: float = 10.0,
    ):
        self.source_embeddings = _as_2d(source_embeddings, "source_embeddings")
        self.target_embeddings = _as_2d(target_embeddings, "target_embeddings")
        self.source_targets = _as_2d(source_targets, "source_targets")
        self.target_targets = _as_2d(target_targets, "target_targets")
        if len(self.source_embeddings) != len(self.source_targets):
            raise ValueError("source_embeddings and source_targets must have equal rows.")
        if len(self.target_embeddings) != len(self.target_targets):
            raise ValueError("target_embeddings and target_targets must have equal rows.")
        self.target_names = target_names or [f"target_{index}" for index in range(self.source_targets.shape[1])]
        self.alpha = float(alpha)
        self.projector = V0FeatureProjector(pooled_shape_size)
        self.source_features = self.projector.fit_transform(self.source_embeddings)
        self.target_features = self.projector.transform(self.target_embeddings)
        self.target_similarity = target_to_source_similarity(
            self.source_embeddings,
            self.target_embeddings,
        )

    def domain_similarity(self) -> dict[str, float]:
        """Summarize target cosine similarity to the source-domain centroid."""
        return {
            "minimum": float(np.min(self.target_similarity)),
            "median": float(np.median(self.target_similarity)),
            "mean": float(np.mean(self.target_similarity)),
            "maximum": float(np.max(self.target_similarity)),
        }

    def target_split(
        self,
        *,
        test_fraction: float = 0.3,
        random_seed: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic target adaptation-pool and held-out indices."""
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        order = np.random.default_rng(random_seed).permutation(len(self.target_features))
        test_count = max(1, int(round(len(order) * test_fraction)))
        return order[test_count:], order[:test_count]

    def fit_models(self, target_indices: Any) -> dict[str, TransferRidgeRegressor]:
        """Fit zero-shot, target-only, and transfer heads for selected target labels."""
        indices = np.asarray(target_indices, dtype=int)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("target_indices must be a non-empty one-dimensional array.")
        if indices.min() < 0 or indices.max() >= len(self.target_features):
            raise IndexError("target_indices contain an out-of-range target row.")
        source_model = TransferRidgeRegressor(self.alpha).fit(
            self.source_features,
            self.source_targets,
        )
        return {
            "zero-shot": source_model,
            "target-only": TransferRidgeRegressor(self.alpha).fit(
                self.target_features[indices],
                self.target_targets[indices],
            ),
            "transfer": TransferRidgeRegressor(self.alpha).fit(
                self.target_features[indices],
                self.target_targets[indices],
                prior=source_model,
            ),
        }

    def learning_curve(
        self,
        sample_sizes: list[int],
        *,
        repeats: int = 12,
        test_fraction: float = 0.3,
        random_seed: int = 15,
    ) -> pd.DataFrame:
        """Run an aggregate source-to-target transfer learning curve."""
        return evaluate_transfer_learning(
            self.source_features,
            self.source_targets,
            self.target_features,
            self.target_targets,
            sample_sizes,
            target_names=self.target_names,
            alpha=self.alpha,
            repeats=repeats,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )

    def similarity_learning_curves(
        self,
        sample_sizes: list[int],
        *,
        bands: int = 3,
        repeats: int = 12,
        test_fraction: float = 0.3,
        random_seed: int = 15,
    ) -> pd.DataFrame:
        """Run separate learning curves for quantile bands of source similarity."""
        if bands < 2:
            raise ValueError("bands must be at least 2.")
        labels = [f"Q{index + 1}" for index in range(bands)]
        assignments = pd.qcut(self.target_similarity, q=bands, labels=labels, duplicates="drop")
        frames = []
        for band_index, label in enumerate(assignments.categories):
            mask = np.asarray(assignments == label)
            available_pool = len(self.target_features[mask]) - max(
                1, int(round(len(self.target_features[mask]) * test_fraction))
            )
            valid_sizes = [size for size in sample_sizes if size <= available_pool]
            if not valid_sizes:
                continue
            frame = evaluate_transfer_learning(
                self.source_features,
                self.source_targets,
                self.target_features[mask],
                self.target_targets[mask],
                valid_sizes,
                target_names=self.target_names,
                alpha=self.alpha,
                repeats=repeats,
                test_fraction=test_fraction,
                random_seed=random_seed + band_index * 1_000,
            )
            frame.insert(0, "similarity_band", str(label))
            frame.insert(1, "similarity_mean", float(np.mean(self.target_similarity[mask])))
            frame.insert(2, "similarity_min", float(np.min(self.target_similarity[mask])))
            frame.insert(3, "similarity_max", float(np.max(self.target_similarity[mask])))
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


class PartitionTransferStudy:
    """Compare one foundation domain with multiple target-domain specialists."""

    def __init__(
        self,
        features: Any,
        targets: Any,
        domains: Any,
        source_domain: Any,
        *,
        target_names: list[str] | None = None,
        alpha: float = 10.0,
    ):
        self.features = _as_2d(features, "features")
        self.targets = _as_2d(targets, "targets")
        self.domains = np.asarray(domains)
        if self.domains.ndim != 1 or len(self.domains) != len(self.features):
            raise ValueError("domains must be one-dimensional and match the feature rows.")
        if len(self.targets) != len(self.features):
            raise ValueError("features and targets must contain the same number of rows.")
        if source_domain not in set(self.domains):
            raise ValueError(f"Unknown source domain: {source_domain!r}.")
        if len(pd.unique(self.domains)) < 2:
            raise ValueError("A partition transfer study requires at least two domains.")
        self.source_domain = source_domain
        self.target_names = target_names or [f"target_{index}" for index in range(self.targets.shape[1])]
        if len(self.target_names) != self.targets.shape[1]:
            raise ValueError("target_names must match the number of target columns.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)

    @property
    def target_domains(self) -> list[Any]:
        """Return target domains in their original encounter order."""
        return [domain for domain in pd.unique(self.domains) if domain != self.source_domain]

    def domain_counts(self) -> pd.DataFrame:
        """Count source and target rows in every domain."""
        counts = pd.Series(self.domains).value_counts(sort=False)
        return pd.DataFrame(
            {
                "domain": counts.index,
                "rows": counts.to_numpy(),
                "role": ["source" if domain == self.source_domain else "target" for domain in counts.index],
            }
        )

    def _domain_arrays(self, target_domain: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if target_domain not in self.target_domains:
            raise ValueError(f"Unknown target domain: {target_domain!r}.")
        source_mask = self.domains == self.source_domain
        target_mask = self.domains == target_domain
        return (
            self.features[source_mask],
            self.targets[source_mask],
            self.features[target_mask],
            self.targets[target_mask],
        )

    @staticmethod
    def _fraction_sizes(
        target_rows: int,
        fractions: list[float],
        test_fraction: float,
    ) -> tuple[list[int], int]:
        if not fractions or any(not 0 < float(fraction) <= 1 for fraction in fractions):
            raise ValueError("fractions must contain values greater than 0 and at most 1.")
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        test_count = max(1, int(round(target_rows * test_fraction)))
        pool_size = target_rows - test_count
        if pool_size < 1:
            raise ValueError("Each target domain must leave at least one adaptation row.")
        sizes = sorted({max(1, min(pool_size, int(round(pool_size * fraction)))) for fraction in fractions})
        return sizes, pool_size

    def learning_curves(
        self,
        fractions: list[float],
        *,
        repeats: int = 12,
        test_fraction: float = 0.3,
        random_seed: int = 16,
    ) -> pd.DataFrame:
        """Evaluate percentage-based transfer curves for every target domain."""
        frames = []
        for domain_index, target_domain in enumerate(self.target_domains):
            x_source, y_source, x_target, y_target = self._domain_arrays(target_domain)
            sizes, pool_size = self._fraction_sizes(len(x_target), fractions, test_fraction)
            frame = evaluate_transfer_learning(
                x_source,
                y_source,
                x_target,
                y_target,
                sizes,
                target_names=self.target_names,
                alpha=self.alpha,
                repeats=repeats,
                test_fraction=test_fraction,
                random_seed=random_seed + 1_000 * domain_index,
            )
            frame.insert(0, "source_domain", self.source_domain)
            frame.insert(1, "target_domain", target_domain)
            frame.insert(2, "sample_fraction", frame["sample_size"] / pool_size)
            frame.insert(3, "target_pool_size", pool_size)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def dedicated_benchmarks(
        self,
        *,
        test_fraction: float = 0.3,
        random_seed: int = 16,
    ) -> pd.DataFrame:
        """Fit each target specialist with its complete adaptation pool."""
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        records = []
        method_names = {
            "zero-shot": "zero-shot",
            "target-only": "dedicated-full",
            "transfer": "transfer-full",
        }
        for domain_index, target_domain in enumerate(self.target_domains):
            x_source, y_source, x_target, y_target = self._domain_arrays(target_domain)
            order = np.random.default_rng(random_seed + 1_000 * domain_index).permutation(len(x_target))
            test_count = max(1, int(round(len(order) * test_fraction)))
            if test_count >= len(order):
                raise ValueError("Each target domain must leave at least one adaptation row.")
            test_indices = order[:test_count]
            pool_indices = order[test_count:]
            source_model = TransferRidgeRegressor(self.alpha).fit(x_source, y_source)
            scores = _evaluate_models(
                source_model,
                x_target[pool_indices],
                y_target[pool_indices],
                x_target[test_indices],
                y_target[test_indices],
                self.alpha,
                self.target_names,
            )
            for method, frame in scores.items():
                for record in frame.to_dict(orient="records"):
                    records.append(
                        {
                            "source_domain": self.source_domain,
                            "target_domain": target_domain,
                            "method": method_names[method],
                            "sample_size": len(pool_indices) if method != "zero-shot" else 0,
                            "sample_fraction": 1.0 if method != "zero-shot" else 0.0,
                            "target_pool_size": len(pool_indices),
                            **record,
                        }
                    )
        return pd.DataFrame(records)


class V0PartitionTransferStudy(PartitionTransferStudy):
    """Run partition transfer studies from a shared v0 embedding catalogue."""

    def __init__(
        self,
        embeddings: Any,
        targets: Any,
        domains: Any,
        source_domain: Any,
        *,
        target_names: list[str] | None = None,
        pooled_shape_size: int = 12,
        alpha: float = 10.0,
    ):
        self.embeddings = _as_2d(embeddings, "embeddings")
        domain_array = np.asarray(domains)
        if domain_array.ndim != 1 or len(domain_array) != len(self.embeddings):
            raise ValueError("domains must be one-dimensional and match the embedding rows.")
        source_mask = domain_array == source_domain
        if not np.any(source_mask):
            raise ValueError(f"Unknown source domain: {source_domain!r}.")
        self.projector = V0FeatureProjector(pooled_shape_size).fit(self.embeddings[source_mask])
        features = self.projector.transform(self.embeddings)
        super().__init__(
            features,
            targets,
            domain_array,
            source_domain,
            target_names=target_names,
            alpha=alpha,
        )

    def domain_similarity(self) -> pd.DataFrame:
        """Summarize each target domain's raw-v0 cosine similarity to the source."""
        source_embeddings = self.embeddings[self.domains == self.source_domain]
        records = []
        for target_domain in self.target_domains:
            similarities = target_to_source_similarity(
                source_embeddings,
                self.embeddings[self.domains == target_domain],
            )
            records.append(
                {
                    "source_domain": self.source_domain,
                    "target_domain": target_domain,
                    "minimum": float(np.min(similarities)),
                    "median": float(np.median(similarities)),
                    "mean": float(np.mean(similarities)),
                    "maximum": float(np.max(similarities)),
                }
            )
        return pd.DataFrame(records)
