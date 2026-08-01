"""Evidence-gated transfer between a source foundation and target specialist."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .adaptation import EBRAAdapter, EBRAConfig
from .model import CapacitancePrediction, TopoCapConfig, TopoCapFoundationModel
from .schema import CapacitanceGraph
from .targets import maxwell_to_components

GraphView = Callable[[CapacitanceGraph], CapacitanceGraph]
GateChoice = Literal["foundation", "transfer", "specialist"]


@dataclass(frozen=True, slots=True)
class EvidenceGateConfig:
    """Settings for support-only model selection without touching target test rows."""

    cross_validation_folds: int = 3
    minimum_specialist_support: int = 8
    minimum_group_count: int = 3
    specialist_relative_margin: float = 0.0
    specialist_standard_error_margin: float = 1.0
    random_seed: int = 73

    def __post_init__(self) -> None:
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least two.")
        if self.minimum_specialist_support < 2:
            raise ValueError("minimum_specialist_support must be at least two.")
        if self.minimum_group_count < 2:
            raise ValueError("minimum_group_count must be at least two.")
        if not 0.0 <= self.specialist_relative_margin < 1.0:
            raise ValueError("specialist_relative_margin must lie in [0, 1).")
        if not np.isfinite(self.specialist_standard_error_margin) or self.specialist_standard_error_margin < 0.0:
            raise ValueError("specialist_standard_error_margin must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class GateEvidence:
    """Auditable support-set evidence behind one fitted gate decision."""

    choice: GateChoice
    support_size: int
    transfer_cv_log_mae: float | None
    specialist_cv_log_mae: float | None
    specialist_paired_improvement: float | None
    specialist_improvement_standard_error: float | None
    fold_count: int
    reason: str


class EvidenceGatedTopoCap:
    """Choose a low-shot source prior or a target-native GDS specialist.

    Selection uses only the labelled target support set.  Below the configured
    minimum support, the lower-capacity transfer path is preferred.  Once a
    target specialist is statistically plausible, grouped cross-validation
    decides between candidates and the winner is refit on every support row.
    """

    def __init__(
        self,
        foundation: TopoCapFoundationModel,
        control_view: GraphView,
        specialist_view: GraphView,
        *,
        adapter_config: EBRAConfig | None = None,
        specialist_config: TopoCapConfig | None = None,
        gate_config: EvidenceGateConfig | None = None,
    ):
        if not foundation.is_fitted:
            raise RuntimeError("Fit the source foundation before constructing an evidence gate.")
        self.foundation = foundation
        self.control_view = control_view
        self.specialist_view = specialist_view
        self.adapter_config = adapter_config or EBRAConfig()
        self.specialist_config = specialist_config or TopoCapConfig(random_feature_dimensions=0, ridge_alpha=30.0)
        self.gate_config = gate_config or EvidenceGateConfig()
        self.adapter_: EBRAAdapter | None = None
        self.specialist_: TopoCapFoundationModel | None = None
        self.evidence_: GateEvidence | None = None

    @property
    def is_fitted(self) -> bool:
        return self.evidence_ is not None

    @property
    def choice(self) -> GateChoice:
        if self.evidence_ is None:
            raise RuntimeError("Fit the evidence gate before reading its choice.")
        return self.evidence_.choice

    def fit(
        self,
        support_samples: Sequence[CapacitanceGraph],
        *,
        groups: Sequence[object] | None = None,
    ) -> EvidenceGatedTopoCap:
        """Select and refit a candidate using only labelled support samples."""
        support = tuple(support_samples)
        if any(not graph.has_target for graph in support):
            raise ValueError("Every support graph must contain a capacitance target.")
        if groups is not None and len(groups) != len(support):
            raise ValueError("groups must contain one label per support graph.")
        self.adapter_ = None
        self.specialist_ = None

        if not support:
            self.evidence_ = GateEvidence(
                choice="foundation",
                support_size=0,
                transfer_cv_log_mae=None,
                specialist_cv_log_mae=None,
                specialist_paired_improvement=None,
                specialist_improvement_standard_error=None,
                fold_count=0,
                reason="No target labels were available; use the source foundation without adaptation.",
            )
            return self

        control_support = tuple(self.control_view(graph) for graph in support)
        if len(support) < self.gate_config.minimum_specialist_support:
            self.adapter_ = EBRAAdapter(self.foundation, self.adapter_config).fit(control_support)
            self.evidence_ = GateEvidence(
                choice="transfer",
                support_size=len(support),
                transfer_cv_log_mae=None,
                specialist_cv_log_mae=None,
                specialist_paired_improvement=None,
                specialist_improvement_standard_error=None,
                fold_count=0,
                reason=(
                    "The support set is below the predeclared specialist evidence threshold; "
                    "retain the lower-capacity source prior."
                ),
            )
            return self

        if groups is not None:
            group_tokens = {repr(value) for value in groups}
            if len(group_tokens) < self.gate_config.minimum_group_count:
                self.adapter_ = EBRAAdapter(self.foundation, self.adapter_config).fit(control_support)
                self.evidence_ = GateEvidence(
                    choice="transfer",
                    support_size=len(support),
                    transfer_cv_log_mae=None,
                    specialist_cv_log_mae=None,
                    specialist_paired_improvement=None,
                    specialist_improvement_standard_error=None,
                    fold_count=0,
                    reason=(
                        "Too few independent support domains were available for grouped model "
                        "selection; retain the lower-capacity source prior."
                    ),
                )
                return self

        folds = _cross_validation_folds(
            len(support),
            groups=groups,
            fold_count=self.gate_config.cross_validation_folds,
            seed=self.gate_config.random_seed,
        )
        transfer_errors: list[float] = []
        specialist_errors: list[float] = []
        all_indices = np.arange(len(support), dtype=np.int64)
        specialist_support = tuple(self.specialist_view(graph) for graph in support)
        for validation_indices in folds:
            training_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
            transfer = EBRAAdapter(self.foundation, self.adapter_config).fit(
                [control_support[index] for index in training_indices]
            )
            specialist = TopoCapFoundationModel(self.specialist_config).fit(
                [specialist_support[index] for index in training_indices]
            )
            transfer_errors.append(
                _prediction_log_mae(transfer, [control_support[index] for index in validation_indices])
            )
            specialist_errors.append(
                _prediction_log_mae(specialist, [specialist_support[index] for index in validation_indices])
            )

        transfer_error = float(np.mean(transfer_errors))
        specialist_error = float(np.mean(specialist_errors))
        paired_improvements = np.asarray(transfer_errors) - np.asarray(specialist_errors)
        paired_improvement = float(np.mean(paired_improvements))
        improvement_standard_error = (
            float(np.std(paired_improvements, ddof=1) / np.sqrt(len(paired_improvements)))
            if len(paired_improvements) > 1
            else float("inf")
        )
        specialist_limit = transfer_error * (1.0 - self.gate_config.specialist_relative_margin)
        improvement_threshold = self.gate_config.specialist_standard_error_margin * improvement_standard_error
        if specialist_error < specialist_limit and paired_improvement > improvement_threshold:
            self.specialist_ = TopoCapFoundationModel(self.specialist_config).fit(specialist_support)
            choice: GateChoice = "specialist"
            reason = (
                "Grouped support-domain cross-validation favored the target-native specialist "
                "by more than the predeclared standard-error margin."
            )
        else:
            self.adapter_ = EBRAAdapter(self.foundation, self.adapter_config).fit(control_support)
            choice = "transfer"
            reason = (
                "Grouped support-domain evidence did not clear the predeclared margin; retain "
                "the source-informed transfer model."
            )
        self.evidence_ = GateEvidence(
            choice=choice,
            support_size=len(support),
            transfer_cv_log_mae=transfer_error,
            specialist_cv_log_mae=specialist_error,
            specialist_paired_improvement=paired_improvement,
            specialist_improvement_standard_error=improvement_standard_error,
            fold_count=len(folds),
            reason=reason,
        )
        return self

    def predict(self, sample: CapacitanceGraph, confidence: float = 0.9) -> CapacitancePrediction:
        """Predict through the selected path while preserving physical reconstruction."""
        if self.evidence_ is None:
            raise RuntimeError("Fit the evidence gate before prediction.")
        if self.evidence_.choice == "foundation":
            return self.foundation.predict(self.control_view(sample), confidence=confidence)
        if self.evidence_.choice == "transfer":
            if self.adapter_ is None:
                raise RuntimeError("The selected transfer adapter is unavailable.")
            return self.adapter_.predict(self.control_view(sample), confidence=confidence)
        if self.specialist_ is None:
            raise RuntimeError("The selected target specialist is unavailable.")
        return self.specialist_.predict(self.specialist_view(sample), confidence=confidence)

    def predict_many(
        self,
        samples: Sequence[CapacitanceGraph],
        confidence: float = 0.9,
    ) -> list[CapacitancePrediction]:
        """Predict a sequence of variable-size capacitance graphs."""
        return [self.predict(sample, confidence=confidence) for sample in samples]


def _cross_validation_folds(
    sample_count: int,
    *,
    groups: Sequence[object] | None,
    fold_count: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    if groups is None:
        units = np.arange(sample_count, dtype=np.int64)
        rng.shuffle(units)
        count = min(fold_count, sample_count)
        return tuple(np.asarray(fold, dtype=np.int64) for fold in np.array_split(units, count) if len(fold))

    labels = np.asarray(groups, dtype=object)
    tokens = np.asarray([repr(value) for value in labels], dtype=object)
    unique = np.unique(tokens)
    if len(unique) < 2:
        raise ValueError("Grouped model selection requires at least two distinct support groups.")
    rng.shuffle(unique)
    count = min(fold_count, len(unique))
    folds = []
    for group_fold in np.array_split(unique, count):
        indices = np.flatnonzero(np.isin(tokens, group_fold))
        if len(indices):
            folds.append(indices.astype(np.int64, copy=False))
    return tuple(folds)


def _prediction_log_mae(
    model: TopoCapFoundationModel | EBRAAdapter,
    samples: Sequence[CapacitanceGraph],
) -> float:
    errors: list[np.ndarray] = []
    for sample in samples:
        expected = maxwell_to_components(sample.capacitance_matrix)
        actual = maxwell_to_components(model.predict(sample).matrix)
        errors.append(np.abs(expected.log_shunts - actual.log_shunts))
        errors.append(np.abs(expected.log_mutuals - actual.log_mutuals))
    return float(np.mean(np.concatenate(errors)))


__all__ = [
    "EvidenceGateConfig",
    "EvidenceGatedTopoCap",
    "GateEvidence",
    "GraphView",
]
