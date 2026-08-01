"""Tool-neutral feature views for transferable TopoCap experiments.

The graph extractor deliberately retains a rich audit representation.  A
transfer study should not automatically expose every descriptor to every
model: solver-domain extents and layout-tool-specific option vocabularies can
otherwise become shortcuts.  This module defines two conservative views:

* a compact topology/control view used by a cross-family foundation model;
* an active-region geometry view used by a target-native specialist.

Raw option names are consumed only by an explicit adapter.  The fitted model
receives canonical dimensions and physical roles, while the exact names stay
in graph metadata so predictions can be mapped back to the originating layout
tool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .datasets import PARAMETER_DIMENSIONS, PARAMETER_FEATURE_NAMES, PARAMETER_ROLES
from .geometry_graph import EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES
from .schema import CapacitanceGraph

Aggregation = Literal["first", "mean", "minimum", "maximum"]


@dataclass(frozen=True, slots=True)
class CanonicalControl:
    """Map one layout-tool knob (or aliases) to a physical control token."""

    name: str
    source_fields: tuple[str, ...]
    dimension: str
    roles: tuple[str, ...]
    aggregation: Aggregation = "first"

    def __post_init__(self) -> None:
        if not self.name or not self.source_fields or any(not field for field in self.source_fields):
            raise ValueError("A canonical control requires a name and at least one source field.")
        if self.dimension not in PARAMETER_DIMENSIONS:
            raise ValueError(f"Unknown parameter dimension: {self.dimension!r}.")
        if not self.roles or any(role not in PARAMETER_ROLES for role in self.roles):
            raise ValueError("Canonical-control roles must be known TopoCap physical roles.")
        if self.aggregation not in {"first", "mean", "minimum", "maximum"}:
            raise ValueError(f"Unknown control aggregation: {self.aggregation!r}.")
        if self.aggregation == "first" and len(self.source_fields) != 1:
            raise ValueError("The 'first' aggregation accepts exactly one source field.")

    def extract(self, values: Mapping[str, float]) -> float:
        """Extract and aggregate this control from normalized numeric options."""
        missing = [field for field in self.source_fields if field not in values]
        if missing:
            raise KeyError(f"Missing source fields for canonical control {self.name!r}: {missing}.")
        observed = np.asarray([values[field] for field in self.source_fields], dtype=np.float64)
        if not np.isfinite(observed).all():
            raise ValueError(f"Canonical control {self.name!r} contains a non-finite value.")
        if self.aggregation == "first":
            return float(observed[0])
        if self.aggregation == "mean":
            return float(np.mean(observed))
        if self.aggregation == "minimum":
            return float(np.min(observed))
        return float(np.max(observed))


@dataclass(frozen=True, slots=True)
class ControlSchema:
    """Explicit feedback adapter from native option names to physical tokens."""

    name: str
    controls: tuple[CanonicalControl, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.controls:
            raise ValueError("A control schema requires a name and at least one control.")
        names = [control.name for control in self.controls]
        if len(set(names)) != len(names):
            raise ValueError("Canonical control names must be unique within a schema.")

    def tokenize(self, graph: CapacitanceGraph) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        """Return canonical values and descriptors without exposing raw names."""
        raw = graph.metadata.get("split_parameter_values")
        if not isinstance(raw, Mapping):
            raise KeyError("Graph metadata must contain normalized split_parameter_values.")
        values = np.asarray([control.extract(raw) for control in self.controls], dtype=np.float64)
        features = np.vstack(
            [
                _canonical_parameter_feature(value, control.dimension, control.roles)
                for value, control in zip(values, self.controls)
            ]
        )
        return tuple(control.name for control in self.controls), values, features


GENERALIZED_NCAP_CONTROLS = ControlSchema(
    name="generalized-ncap-active-region-v1",
    controls=(
        CanonicalControl("active_count", ("finger_count",), "count", ("count", "active_coupling_region")),
        CanonicalControl("active_length_um", ("finger_length",), "length_um", ("length", "active_coupling_region")),
        CanonicalControl("active_width_um", ("finger_width",), "length_um", ("width", "active_coupling_region")),
        CanonicalControl(
            "active_gap_um",
            ("finger_gap_east_west", "finger_gap_north_south"),
            "length_um",
            ("gap", "active_coupling_region"),
            aggregation="mean",
        ),
    ),
)

CAPN_INTERDIGITAL_TEE_CONTROLS = ControlSchema(
    name="capn-interdigital-tee-active-region-v1",
    controls=(
        CanonicalControl("active_count", ("finger_count",), "count", ("count", "active_coupling_region")),
        CanonicalControl("active_length_um", ("finger_length",), "length_um", ("length", "active_coupling_region")),
        CanonicalControl("active_width_um", ("cap_width",), "length_um", ("width", "active_coupling_region")),
        CanonicalControl("active_gap_um", ("cap_gap",), "length_um", ("gap", "active_coupling_region")),
    ),
)

_CONTROL_NODE_FEATURES = ("is_reference",)
_CONTROL_EDGE_FEATURES = ("reference_incidence",)
_CONTROL_GLOBAL_FEATURES = ("net_count", "reference_count")

ACTIVE_NODE_FEATURES = (
    "is_reference",
    "area_um2",
    "perimeter_um",
    "bbox_width_um",
    "bbox_height_um",
    "bbox_aspect_ratio",
    "bbox_occupancy",
    "central_mu20_um2",
    "central_mu02_um2",
    "central_abs_mu11_um2",
    "polygon_count",
    "hole_count",
    "compactness",
    "solidity",
    "min_distance_to_reference_um",
    "median_boundary_distance_to_reference_um",
)

ACTIVE_EDGE_FEATURES = (
    "minimum_separation_um",
    "separation_q10_um",
    "separation_q25_um",
    "separation_q50_um",
    "separation_q75_um",
    "separation_q90_um",
    "centroid_abs_dx_um",
    "centroid_abs_dy_um",
    "centroid_distance_um",
    "bbox_x_overlap_um",
    "bbox_y_overlap_um",
    "bbox_x_overlap_fraction",
    "bbox_y_overlap_fraction",
    "boundary_orientation_cosine",
    "boundary_orientation_overlap",
    "boundary_orientation_l1_similarity",
    "absolute_log_area_ratio",
    "absolute_log_perimeter_ratio",
    "reference_incidence",
    *(name for name in EDGE_FEATURE_NAMES if name.startswith("proximity_")),
)

ACTIVE_GLOBAL_FEATURES = (
    "net_count",
    "reference_count",
    "layout_width_um",
    "layout_height_um",
    "layout_bbox_area_um2",
    "total_net_area_um2",
    "total_net_perimeter_um",
)


def ncap_control_schema(graph: CapacitanceGraph) -> ControlSchema:
    """Resolve the explicit compatibility adapter for the two NCap releases."""
    family = graph.metadata.get("dataset_family")
    if family == "GeneralizedCapNInterdigital":
        return GENERALIZED_NCAP_CONTROLS
    if family == "CapNInterdigitalTee":
        return CAPN_INTERDIGITAL_TEE_CONTROLS
    raise KeyError(
        f"No built-in NCap control schema for {family!r}; pass an explicit ControlSchema for this layout tool."
    )


def build_topology_control_view(
    graph: CapacitanceGraph,
    schema: ControlSchema | None = None,
) -> CapacitanceGraph:
    """Build the low-capacity, topology-aware cross-family foundation view."""
    selected = schema or ncap_control_schema(graph)
    return _build_view(
        graph,
        node_names=_CONTROL_NODE_FEATURES,
        edge_names=_CONTROL_EDGE_FEATURES,
        global_names=_CONTROL_GLOBAL_FEATURES,
        control_schema=selected,
        view_name="topology-control",
    )


def build_active_geometry_view(
    graph: CapacitanceGraph,
    schema: ControlSchema | None = None,
    *,
    include_controls: bool = True,
) -> CapacitanceGraph:
    """Build the local GDS view used by a target-native geometry specialist."""
    selected = (schema or ncap_control_schema(graph)) if include_controls else None
    return _build_view(
        graph,
        node_names=ACTIVE_NODE_FEATURES,
        edge_names=ACTIVE_EDGE_FEATURES,
        global_names=ACTIVE_GLOBAL_FEATURES,
        control_schema=selected,
        view_name="active-geometry-controls" if include_controls else "active-geometry-only",
    )


def _canonical_parameter_feature(value: float, dimension: str, roles: Sequence[str]) -> np.ndarray:
    row = np.zeros(len(PARAMETER_FEATURE_NAMES), dtype=np.float64)
    row[PARAMETER_DIMENSIONS.index(dimension)] = 1.0
    role_offset = len(PARAMETER_DIMENSIONS) + 1
    gated_offset = role_offset + len(PARAMETER_ROLES)
    for role in roles:
        index = PARAMETER_ROLES.index(role)
        row[role_offset + index] = 1.0
        row[gated_offset + index] = value
    return row


def _indices(all_names: Sequence[str], selected: Sequence[str]) -> list[int]:
    return [all_names.index(name) for name in selected]


def _build_view(
    graph: CapacitanceGraph,
    *,
    node_names: Sequence[str],
    edge_names: Sequence[str],
    global_names: Sequence[str],
    control_schema: ControlSchema | None,
    view_name: str,
) -> CapacitanceGraph:
    if control_schema is None:
        parameter_names: tuple[str, ...] = ()
        parameter_values = np.empty(0, dtype=np.float64)
        parameter_features = np.empty((0, len(PARAMETER_FEATURE_NAMES)), dtype=np.float64)
        schema_name = None
    else:
        parameter_names, parameter_values, parameter_features = control_schema.tokenize(graph)
        schema_name = control_schema.name
    metadata = dict(graph.metadata)
    metadata["topocap_view"] = {
        "name": view_name,
        "control_schema": schema_name,
        "node_features": list(node_names),
        "edge_features": list(edge_names),
        "global_features": list(global_names),
    }
    return CapacitanceGraph(
        node_features=graph.node_features[:, _indices(NODE_FEATURE_NAMES, node_names)],
        edge_index=graph.edge_index,
        edge_features=graph.edge_features[:, _indices(EDGE_FEATURE_NAMES, edge_names)],
        global_features=graph.global_features[_indices(GLOBAL_FEATURE_NAMES, global_names)],
        parameter_values=parameter_values,
        parameter_features=parameter_features,
        net_ids=graph.net_ids,
        parameter_names=parameter_names,
        capacitance_matrix=graph.capacitance_matrix,
        metadata=metadata,
    )


__all__ = [
    "ACTIVE_EDGE_FEATURES",
    "ACTIVE_GLOBAL_FEATURES",
    "ACTIVE_NODE_FEATURES",
    "CAPN_INTERDIGITAL_TEE_CONTROLS",
    "CanonicalControl",
    "ControlSchema",
    "GENERALIZED_NCAP_CONTROLS",
    "build_active_geometry_view",
    "build_topology_control_view",
    "ncap_control_schema",
]
