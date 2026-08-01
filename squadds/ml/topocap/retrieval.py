"""Frozen support-conditioned source retrieval for TopoCap transfer studies.

The retriever is intentionally narrow: it is fitted from source control views
and queried with labelled or unlabelled target support views. It has no target
catalogue or test-set input, and it never reads capacitance targets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .schema import CapacitanceGraph

RETRIEVAL_PROTOCOL_VERSION: Final = "topocap-support-retrieval-1.0.0"
RETRIEVAL_SOURCE_BUDGET: Final = 2_048
RETRIEVAL_SHUFFLE_SEED_XOR: Final = 20_260_801
RETRIEVAL_CONTROL_NAMES: Final = (
    "active_count",
    "active_length_um",
    "active_width_um",
    "active_gap_um",
)
RETRIEVAL_DISTANCE_CONTROL_NAMES: Final = RETRIEVAL_CONTROL_NAMES[1:]
_MINIMUM_SCALE: Final = 1.0e-8
_STANDARDIZED_CLIP: Final = 8.0


def retrieval_source_ids_sha256(source_ids: Sequence[str]) -> str:
    """Hash an ordered source-ID selection using canonical JSON bytes."""
    identifiers = tuple(str(source_id) for source_id in source_ids)
    payload = json.dumps(
        identifiers,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SupportRetrievalSelection:
    """Immutable indices and provenance for one retrieved source subset."""

    source_indices: tuple[int, ...]
    source_ids: tuple[str, ...]
    source_ids_sha256: str
    source_budget: int = RETRIEVAL_SOURCE_BUDGET
    protocol_version: str = RETRIEVAL_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.source_budget != RETRIEVAL_SOURCE_BUDGET:
            raise ValueError(f"source_budget must remain frozen at {RETRIEVAL_SOURCE_BUDGET}.")
        if self.protocol_version != RETRIEVAL_PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {RETRIEVAL_PROTOCOL_VERSION!r}.")
        if len(self.source_indices) != self.source_budget or len(self.source_ids) != self.source_budget:
            raise ValueError("A retrieval selection must contain exactly the frozen source budget.")
        if len(set(self.source_indices)) != len(self.source_indices):
            raise ValueError("source_indices must be unique.")
        if any(index < 0 for index in self.source_indices):
            raise ValueError("source_indices must be non-negative.")
        if len(set(self.source_ids)) != len(self.source_ids) or any(not source_id for source_id in self.source_ids):
            raise ValueError("source_ids must be non-empty and unique.")
        if self.source_ids_sha256 != retrieval_source_ids_sha256(self.source_ids):
            raise ValueError("source_ids_sha256 does not match the ordered source IDs.")


class SupportConditionedSourceRetriever:
    """Retrieve a fixed source budget using target-support geometry only.

    Distances use ``log1p`` transformed canonical active length, width, and gap.
    The transform is normalized by source-only medians and interquartile ranges.
    Selection is equally stratified over observed source finger counts, with a
    deterministic global fill only when a stratum cannot meet its quota.
    """

    def __init__(self, source_graphs: Sequence[CapacitanceGraph]):
        source = tuple(source_graphs)
        if len(source) < RETRIEVAL_SOURCE_BUDGET:
            raise ValueError(f"At least {RETRIEVAL_SOURCE_BUDGET} source graphs are required; received {len(source)}.")
        source_values = _control_values(source, role="source")
        source_ids = tuple(_source_id(graph, index) for index, graph in enumerate(source))
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Source graph source_id values must be unique.")

        transformed = np.log1p(np.maximum(source_values[:, 1:4], 0.0))
        center = np.median(transformed, axis=0)
        q25, q75 = np.quantile(transformed, [0.25, 0.75], axis=0)
        scale = q75 - q25
        scale[scale < _MINIMUM_SCALE] = 1.0
        coordinates = np.clip((transformed - center) / scale, -_STANDARDIZED_CLIP, _STANDARDIZED_CLIP)

        self._source_ids = source_ids
        self._source_coordinates = _readonly(coordinates)
        self._source_counts = _readonly(np.rint(source_values[:, 0]).astype(np.int64))
        self._center = _readonly(center)
        self._scale = _readonly(scale)

    @property
    def source_count(self) -> int:
        """Return the number of source records available to the retriever."""
        return len(self._source_ids)

    @property
    def source_center(self) -> NDArray[np.float64]:
        """Return a copy of the frozen source-only median vector."""
        return self._center.copy()

    @property
    def source_scale(self) -> NDArray[np.float64]:
        """Return a copy of the frozen source-only interquartile range vector."""
        return self._scale.copy()

    @staticmethod
    def protocol() -> dict[str, object]:
        """Return the immutable protocol fields bound into study state."""
        return {
            "version": RETRIEVAL_PROTOCOL_VERSION,
            "source_budget": RETRIEVAL_SOURCE_BUDGET,
            "shuffle_seed_xor": RETRIEVAL_SHUFFLE_SEED_XOR,
            "distance_controls": list(RETRIEVAL_DISTANCE_CONTROL_NAMES),
            "transform": "log1p(max(value, 0))",
            "normalization": "source-only median/IQR",
            "standardized_clip": _STANDARDIZED_CLIP,
            "stratification": "equal quota across sorted observed source active_count values",
            "test_features_or_labels_used": False,
        }

    def retrieve(self, support_graphs: Sequence[CapacitanceGraph]) -> SupportRetrievalSelection:
        """Select source rows from support controls without reading any target label."""
        support = tuple(support_graphs)
        if not support:
            raise ValueError("At least one support graph is required; K=0 has no retrieval fallback.")
        support_values = _control_values(support, role="support")
        transformed = np.log1p(np.maximum(support_values[:, 1:4], 0.0))
        support_coordinates = np.clip(
            (transformed - self._center) / self._scale,
            -_STANDARDIZED_CLIP,
            _STANDARDIZED_CLIP,
        )
        squared = np.sum(
            (self._source_coordinates[:, None, :] - support_coordinates[None, :, :]) ** 2,
            axis=2,
        )
        distance = np.sqrt(np.min(squared, axis=1))
        selected = self._stratified_indices(distance)
        source_ids = tuple(self._source_ids[index] for index in selected)
        return SupportRetrievalSelection(
            source_indices=tuple(map(int, selected)),
            source_ids=source_ids,
            source_ids_sha256=retrieval_source_ids_sha256(source_ids),
        )

    def _stratified_indices(self, distance: NDArray[np.float64]) -> NDArray[np.int64]:
        unique_counts = np.unique(self._source_counts)
        base, remainder = divmod(RETRIEVAL_SOURCE_BUDGET, len(unique_counts))
        selected: list[int] = []
        for position, count in enumerate(unique_counts):
            candidates = np.flatnonzero(self._source_counts == count)
            quota = min(len(candidates), base + int(position < remainder))
            order = np.lexsort((candidates, distance[candidates]))
            selected.extend(map(int, candidates[order[:quota]]))

        if len(selected) < RETRIEVAL_SOURCE_BUDGET:
            chosen = set(selected)
            remaining = np.asarray(
                [index for index in range(self.source_count) if index not in chosen],
                dtype=np.int64,
            )
            order = np.lexsort((remaining, distance[remaining]))
            selected.extend(map(int, remaining[order[: RETRIEVAL_SOURCE_BUDGET - len(selected)]]))

        result = np.asarray(sorted(set(selected)), dtype=np.int64)
        if len(result) != RETRIEVAL_SOURCE_BUDGET:
            raise RuntimeError(f"Retrieval produced {len(result)} unique rows; expected {RETRIEVAL_SOURCE_BUDGET}.")
        return result


def retrieve_support_conditioned_source(
    source_graphs: Sequence[CapacitanceGraph],
    support_graphs: Sequence[CapacitanceGraph],
) -> SupportRetrievalSelection:
    """Run the frozen retriever in one call."""
    return SupportConditionedSourceRetriever(source_graphs).retrieve(support_graphs)


def _control_values(graphs: Sequence[CapacitanceGraph], *, role: str) -> NDArray[np.float64]:
    rows = []
    for index, graph in enumerate(graphs):
        if tuple(graph.parameter_names) != RETRIEVAL_CONTROL_NAMES:
            raise ValueError(
                f"{role} graph {index} must be a topology-control view with parameter names "
                f"{RETRIEVAL_CONTROL_NAMES!r}."
            )
        values = np.asarray(graph.parameter_values, dtype=np.float64)
        if values.shape != (len(RETRIEVAL_CONTROL_NAMES),) or not np.isfinite(values).all():
            raise ValueError(f"{role} graph {index} has invalid canonical control values.")
        rows.append(values)
    return np.vstack(rows)


def _source_id(graph: CapacitanceGraph, index: int) -> str:
    source_id = graph.metadata.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"Source graph {index} requires a non-empty metadata source_id.")
    return source_id


def _readonly(values: NDArray) -> NDArray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "RETRIEVAL_CONTROL_NAMES",
    "RETRIEVAL_DISTANCE_CONTROL_NAMES",
    "RETRIEVAL_PROTOCOL_VERSION",
    "RETRIEVAL_SHUFFLE_SEED_XOR",
    "RETRIEVAL_SOURCE_BUDGET",
    "SupportConditionedSourceRetriever",
    "SupportRetrievalSelection",
    "retrieval_source_ids_sha256",
    "retrieve_support_conditioned_source",
]
