"""Deterministic synthetic fixtures shared by the TopoCap tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from squadds.ml.topocap.geometry_graph import EDGE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES
from squadds.ml.topocap.schema import CapacitanceGraph, canonical_edge_index
from squadds.ml.topocap.targets import components_to_maxwell, maxwell_to_components


def physical_matrix(node_count: int, *, scale: float = 1.0) -> np.ndarray:
    """Return a strictly physical, variable-size signed Maxwell matrix."""
    edges = canonical_edge_index(node_count)
    shunts = scale * (1.2 + 0.08 * np.arange(node_count, dtype=float))
    separation = edges[1] - edges[0]
    mutuals = scale * (0.12 + 0.5 / (1.0 + separation))
    return components_to_maxwell(shunts, mutuals, edges)


def synthetic_graph(
    node_count: int,
    *,
    geometry_scale: float = 1.0,
    target_scale: float = 1.0,
    family: str = "synthetic",
) -> CapacitanceGraph:
    """Build a complete graph whose descriptors are symmetric and nontrivial."""
    positions = np.linspace(-1.0, 1.0, node_count)
    node_features = np.column_stack(
        (
            geometry_scale * (1.0 + positions**2),
            np.sin(np.pi * positions),
            np.cos(np.pi * positions),
        )
    )
    edges = canonical_edge_index(node_count)
    first = node_features[edges[0]]
    second = node_features[edges[1]]
    edge_features = np.column_stack(
        (
            np.abs(positions[edges[0]] - positions[edges[1]]),
            first[:, 0] + second[:, 0],
            np.abs(first[:, 1] - second[:, 1]),
            first[:, 2] * second[:, 2],
        )
    )
    parameter_values = np.asarray([geometry_scale, float(node_count)])
    parameter_features = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    return CapacitanceGraph(
        node_features=node_features,
        edge_index=edges,
        edge_features=edge_features,
        global_features=np.asarray([np.log1p(node_count), geometry_scale]),
        parameter_values=parameter_values,
        parameter_features=parameter_features,
        parameter_names=("geometry_scale", "conductor_count"),
        net_ids=tuple(f"net_{index:03d}" for index in range(node_count)),
        capacitance_matrix=physical_matrix(
            node_count,
            scale=target_scale * (0.8 + 0.2 * geometry_scale),
        ),
        metadata={"dataset_family": family},
    )


def rescale_target(
    graph: CapacitanceGraph,
    *,
    shunt_factor: float,
    mutual_factor: float,
) -> CapacitanceGraph:
    """Return the same design with a controlled target-domain residual."""
    components = maxwell_to_components(graph.capacitance_matrix)
    matrix = components_to_maxwell(
        components.shunts * shunt_factor,
        components.mutuals * mutual_factor,
        components.edge_index,
    )
    return graph.with_target(matrix)


def synthetic_view_graph(
    node_count: int,
    *,
    family: str = "GeneralizedCapNInterdigital",
    active_count: float = 6.0,
    active_length_um: float = 20.0,
    active_width_um: float = 2.0,
    active_gap_um: float = 1.0,
    geometry_signal: float = 1.0,
    target_kind: str = "control",
) -> CapacitanceGraph:
    """Build a full-width graph accepted by the canonical TopoCap views."""
    positions = np.linspace(-1.0, 1.0, node_count)
    reference = np.zeros(node_count, dtype=float)
    reference[0] = 1.0
    node_features = np.empty((node_count, len(NODE_FEATURE_NAMES)), dtype=float)
    for column, name in enumerate(NODE_FEATURE_NAMES):
        if name == "is_reference":
            values = reference
        elif name in {"polygon_count", "hole_count", "port_count"}:
            values = (column % 3) + (1.0 - reference)
        elif name.startswith("boundary_orientation_bin_"):
            values = 0.1 + 0.02 * column + 0.01 * np.abs(positions)
        else:
            values = 0.05 * (column + 1) + geometry_signal * (1.0 + 0.2 * np.abs(positions))
        node_features[:, column] = values

    edges = canonical_edge_index(node_count)
    edge_distance = np.abs(positions[edges[0]] - positions[edges[1]])
    reference_incidence = reference[edges[0]] + reference[edges[1]]
    edge_features = np.empty((edges.shape[1], len(EDGE_FEATURE_NAMES)), dtype=float)
    for column, name in enumerate(EDGE_FEATURE_NAMES):
        if name == "reference_incidence":
            values = reference_incidence
        elif name.startswith("proximity_mean_"):
            values = np.exp(-edge_distance * (1.0 + 0.01 * column))
        elif name.startswith("proximity_length_"):
            values = (1.0 + geometry_signal) * np.exp(-edge_distance * (1.0 + 0.01 * column))
        elif name.startswith("bbox_") or "orientation" in name:
            values = 0.1 * (column + 1) + 0.05 * edge_distance
        else:
            values = 0.03 * (column + 1) + edge_distance + 0.25 * geometry_signal
        edge_features[:, column] = values

    global_features = np.empty(len(GLOBAL_FEATURE_NAMES), dtype=float)
    for column, name in enumerate(GLOBAL_FEATURE_NAMES):
        if name == "net_count":
            value = float(node_count)
        elif name == "reference_count":
            value = 1.0
        else:
            value = 0.2 * (column + 1) + geometry_signal * node_count
        global_features[column] = value

    if family == "GeneralizedCapNInterdigital":
        split_parameters = {
            "finger_count": active_count,
            "finger_length": active_length_um,
            "finger_width": active_width_um,
            "finger_gap_east_west": active_gap_um * 0.8,
            "finger_gap_north_south": active_gap_um * 1.2,
        }
    elif family == "CapNInterdigitalTee":
        split_parameters = {
            "finger_count": active_count,
            "finger_length": active_length_um,
            "cap_width": active_width_um,
            "cap_gap": active_gap_um,
        }
    else:
        split_parameters = {
            "native_count": active_count,
            "native_length": active_length_um,
            "native_width": active_width_um,
            "native_gap": active_gap_um,
        }

    if target_kind == "control":
        log_scale = 0.035 * active_count + 0.012 * active_length_um + 0.08 * active_width_um - 0.06 * active_gap_um
    elif target_kind == "geometry":
        log_scale = 0.65 * geometry_signal
    else:
        raise ValueError(f"Unknown target_kind: {target_kind!r}.")

    return CapacitanceGraph(
        node_features=node_features,
        edge_index=edges,
        edge_features=edge_features,
        global_features=global_features,
        net_ids=tuple(f"net_{index:03d}" for index in range(node_count)),
        capacitance_matrix=physical_matrix(node_count, scale=float(np.exp(log_scale))),
        metadata={
            "dataset_family": family,
            "split_parameter_values": split_parameters,
            "raw_parameter_values": {f"{family}.raw.private_knob": 123.0},
        },
    )


def write_generalized_test_gds(path: Path) -> Path:
    """Write a minimal GeneralizedNCap-like GDS with explicit port markers."""
    import gdstk

    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("TOP")

    for rectangle in (
        ((-5.0, -5.0), (50.0, 0.0)),
        ((-5.0, 20.0), (50.0, 25.0)),
        ((-5.0, 0.0), (0.0, 20.0)),
        ((45.0, 0.0), (50.0, 20.0)),
    ):
        cell.add(gdstk.rectangle(*rectangle, layer=1, datatype=0))

    cell.add(gdstk.rectangle((5.0, 5.0), (15.0, 15.0), layer=1, datatype=10))
    cell.add(gdstk.rectangle((30.0, 5.0), (40.0, 15.0), layer=1, datatype=10))
    cell.add(gdstk.rectangle((8.0, 8.0), (9.0, 9.0), layer=2, datatype=0))
    cell.add(gdstk.rectangle((36.0, 8.0), (37.0, 9.0), layer=3, datatype=0))
    library.write_gds(str(path))
    return path


def generalized_row(matrix_ff: np.ndarray | None = None) -> dict:
    """Return one explicit-unit GeneralizedNCap simulation row."""
    matrix = physical_matrix(3, scale=2.0) if matrix_ff is None else np.asarray(matrix_ff)
    labels = ("G", "N", "S")
    matrix_values = {
        f"C_{row}_{column}": float(matrix[row_index, column_index])
        for row_index, row in enumerate(labels)
        for column_index, column in enumerate(labels)
    }
    return {
        "design": {
            "design_options": {
                "finger_count": "5",
                "finger_length": "20um",
                "finger_width": "2um",
                "finger_gap": "1um",
            }
        },
        "notes": {
            "source_id": "exp6/cap_0001",
            "source_campaign": "exp6",
            "source_file": "exp6/cap_0001.json",
        },
        "sim_results": {
            "units": "fF",
            "maxwell_matrix": matrix_values,
        },
    }
