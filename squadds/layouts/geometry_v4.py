"""Arbitrary-terminal, rigid-motion-invariant planar layout graph.

``planar-terminal-graph-v4`` is an additive experimental successor to v3.  The
canonical representation is a ragged graph because no finite pooled vector can
retain an unbounded number of conductors without loss.  Fixed-width layout and
pair-query views are deterministic derivatives for visualization and ordinary
regressors.

The implementation is intentionally planar.  Its layer-contract parser already
uses a versioned intermediate representation so a future multi-plane encoder can
attach physical z coordinates, thicknesses, and materials without redefining
the planar geometry channels.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry_v2 import VACUUM_PERMITTIVITY, read_layer_geometry, soft_histogram
from .geometry_v3 import (
    V3_BLOCK_SLICES,
    _component_and_hole_count,
    _operator_values,
    _polygon_parts,
    _shape_lines,
    encode_v3,
)

PLANAR_TERMINAL_GRAPH_V4_MODEL = "planar-terminal-graph-v4"
PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION = "4.0.0-experimental"

V4_NODE_DIMENSIONS = 64
V4_EDGE_DIMENSIONS = 96
V4_GLOBAL_DIMENSIONS = 64
V4_PAIR_VIEW_DIMENSIONS = 480
V4_LAYOUT_VIEW_DIMENSIONS = 544
V4_HYBRID_PAIR_DIMENSIONS = 432

V4_NORMALIZED_PAIR_EDGES = np.logspace(-3.0, 2.0, 25)
V4_ABSOLUTE_PAIR_EDGES_UM = np.logspace(-1.0, 3.0, 25)
V4_NORMALIZED_SELF_EDGES = np.logspace(-3.0, 1.0, 17)
V4_NORMALIZED_GROUND_EDGES = np.logspace(-3.0, 2.0, 13)
V4_ABSOLUTE_GROUND_EDGES_UM = np.logspace(-1.0, 3.0, 13)
V4_TOPOLOGY_RADII_NORMALIZED = np.logspace(-3.0, 0.0, 8)


@dataclass(frozen=True)
class PlanarGraphV4:
    """Permutation-equivariant geometry and operator state for one layout."""

    node_features: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    node_ids: tuple[str, ...]
    node_roles: tuple[str, ...]
    queryable_mask: np.ndarray
    proxy_matrix_fF: np.ndarray
    metadata: dict[str, Any]


def _require_shapely():
    try:
        import shapely
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("planar-terminal-graph-v4 requires shapely and the SQuADDS gds extra.") from exc
    return shapely


def _load_contract(
    geometry: Mapping[tuple[int, int], Any],
    layer_contract: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if layer_contract is None:
        layers = []
        for layer, datatype in sorted(geometry):
            if (layer, datatype) == (1, 0):
                role = "ground"
            elif (layer, datatype) == (1, 10):
                role = "conductor"
            elif (layer, datatype) == (1, 11):
                role = "etch"
            elif datatype == 0 and layer >= 2:
                role = "port"
            else:
                role = "ignored"
            entry: dict[str, Any] = {"layer": layer, "datatype": datatype, "role": role}
            if role == "port":
                entry["terminal_index"] = layer - 2
                entry["terminal_id"] = f"t{layer - 2}"
                entry["net_id"] = entry["terminal_id"]
            layers.append(entry)
        return {
            "schema_version": "inferred-planar-v4",
            "units": "um",
            "planes": [{"plane_id": "device", "z_um": 0.0}],
            "layers": layers,
        }
    if isinstance(layer_contract, (str, Path)):
        contract = json.loads(Path(layer_contract).read_text())
    else:
        contract = dict(layer_contract)
    if contract.get("units", "um") != "um":
        raise ValueError("v4 currently requires a layer contract expressed in micrometres.")
    planes = contract.get("planes")
    if planes:
        z_values = {float(plane.get("z_um", 0.0)) for plane in planes}
        if len(z_values) != 1:
            raise ValueError("planar-terminal-graph-v4 accepts exactly one physical conductor plane.")
    return contract


def _contract_roles(contract: Mapping[str, Any]) -> tuple[dict[str, list[tuple[int, int]]], dict[tuple[int, int], dict[str, Any]]]:
    roles = {"conductor": [], "ground": [], "etch": [], "port": []}
    entries: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in contract.get("layers", []):
        entry = dict(raw)
        key = (int(entry["layer"]), int(entry.get("datatype", 0)))
        role = str(entry.get("role", "ignored"))
        if role == "domain":
            role = "ground"
        entries[key] = entry
        if role in roles:
            roles[role].append(key)
    if not roles["conductor"]:
        raise ValueError("The v4 layer contract declares no conductor layers.")
    return roles, entries


def _union_layers(geometry: Mapping[tuple[int, int], Any], keys: Sequence[tuple[int, int]]) -> Any | None:
    shapely = _require_shapely()
    shapes = [geometry[key] for key in keys if key in geometry and not geometry[key].is_empty]
    return shapely.union_all(shapes) if shapes else None


def _local_ground_lines(ground: Any | None) -> list[Any]:
    """Return only moat-facing interior rings; the outer crop is never physics."""
    if ground is None:
        return []
    shapely = _require_shapely()
    lines = []
    for polygon in _polygon_parts(ground):
        lines.extend(shapely.LineString(ring.coords) for ring in polygon.interiors)
    return lines


def _boundary_quadrature(shape_or_lines: Any | Sequence[Any], target: int = 64) -> tuple[np.ndarray, np.ndarray]:
    shapely = _require_shapely()
    if isinstance(shape_or_lines, Sequence) and not hasattr(shape_or_lines, "geom_type"):
        lines = list(shape_or_lines)
    else:
        lines = _shape_lines(shape_or_lines)
    lines = [line for line in lines if float(line.length) > 1e-12]
    if not lines:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    total = sum(float(line.length) for line in lines)
    points = []
    weights = []
    for line in lines:
        count = max(4, int(round(target * float(line.length) / total)))
        positions = (np.arange(count, dtype=np.float64) + 0.5) * float(line.length) / count
        sampled = shapely.line_interpolate_point(line, positions)
        points.append(shapely.get_coordinates(sampled))
        weights.append(np.full(count, float(line.length) / count, dtype=np.float64))
    return np.vstack(points), np.concatenate(weights)


def _pair_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    delta = first[:, None, :] - second[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not len(values) or float(np.sum(weights)) <= 0:
        return 0.0
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, probability * cumulative[-1], side="left")), len(values) - 1)
    return float(values[order[index]])


def _covariance_invariants(shape: Any, scale: float) -> tuple[float, float, float]:
    points, weights = _boundary_quadrature(shape, 96)
    if not len(points):
        return 0.0, 0.0, 0.0
    weights = weights / max(float(weights.sum()), 1e-12)
    center = np.sum(points * weights[:, None], axis=0)
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    eigenvalues = np.maximum(np.linalg.eigvalsh(0.5 * (covariance + covariance.T)), 0.0)
    eccentricity = math.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[-1], 1e-12)))
    return eccentricity, math.sqrt(eigenvalues[0]) / max(scale, 1e-12), math.sqrt(eigenvalues[-1]) / max(scale, 1e-12)


def _terminal_nodes(
    conductor: Any,
    geometry: Mapping[tuple[int, int], Any],
    port_keys: Sequence[tuple[int, int]],
    entries: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[list[Any], list[str], list[str], np.ndarray, list[int]]:
    merged = conductor.buffer(0)
    parts = [part for part in getattr(merged, "geoms", [merged]) if part.geom_type == "Polygon" and part.area > 0]
    if not parts:
        raise ValueError("Conductor geometry contains no polygonal components.")
    tolerance = 2.0e-3  # two database units for the 1-nm standard
    assignments: dict[int, list[tuple[int, str, str]]] = {index: [] for index in range(len(parts))}
    marker_count = 0
    for fallback, key in enumerate(sorted(port_keys)):
        port = geometry.get(key)
        if port is None or port.is_empty:
            continue
        distances = np.asarray([float(part.distance(port)) for part in parts])
        touching = np.flatnonzero(distances <= tolerance)
        entry = entries.get(key, {})
        if len(touching) != 1:
            # Nested or tightly spaced conductors can make a 2-D marker cross
            # more than one island even though its semantic terminal binding is
            # unambiguous.  A sidecar may therefore identify the intended
            # conductor by its physical area/perimeter.  Geometry-only inputs
            # remain strict and fail rather than guessing.
            expected_area = entry.get("conductor_area_um2")
            expected_perimeter = entry.get("conductor_perimeter_um")
            if expected_area is None or expected_perimeter is None:
                raise ValueError(f"Port layer {key} touches {len(touching)} conductor components; expected exactly one.")
            scores = np.asarray(
                [
                    abs(float(part.area) - float(expected_area)) / max(abs(float(expected_area)), 1e-12)
                    + abs(float(part.length) - float(expected_perimeter))
                    / max(abs(float(expected_perimeter)), 1e-12)
                    for part in parts
                ],
                dtype=np.float64,
            )
            order = np.argsort(scores, kind="stable")
            if scores[order[0]] > 5e-3 or (len(order) > 1 and scores[order[1]] - scores[order[0]] <= 1e-8):
                raise ValueError(f"Port layer {key} has no unique conductor match in its semantic sidecar.")
            touching = np.asarray([int(order[0])])
        terminal_index = int(entry.get("terminal_index", fallback))
        terminal_id = str(entry.get("terminal_id", f"t{terminal_index}"))
        net_id = str(entry.get("net_id", terminal_id))
        assignments[int(touching[0])].append((terminal_index, terminal_id, net_id))
        marker_count += 1

    terminal_rows = []
    floating_rows = []
    for part_index, part in enumerate(parts):
        assigned = assignments[part_index]
        if assigned:
            net_ids = {item[2] for item in assigned}
            if len(net_ids) != 1:
                raise ValueError(f"Conductor component {part_index} is assigned to conflicting nets: {sorted(net_ids)}")
            terminal_index, terminal_id, _ = min(assigned)
            terminal_rows.append((terminal_index, terminal_id, part, len(assigned)))
        else:
            signature = (-float(part.area), -float(part.length), part_index)
            floating_rows.append((signature, part))
    terminal_rows.sort(key=lambda item: (item[0], item[1]))
    if len({item[1] for item in terminal_rows}) != len(terminal_rows):
        raise ValueError("Terminal IDs must be unique across galvanic conductor components.")
    floating_rows.sort(key=lambda item: item[0])

    nodes = [item[2] for item in terminal_rows] + [item[1] for item in floating_rows]
    node_ids = [item[1] for item in terminal_rows] + [f"f{index}" for index in range(len(floating_rows))]
    node_roles = ["terminal"] * len(terminal_rows) + ["unmarked"] * len(floating_rows)
    # Preserve every disconnected conductor as a capacitance-matrix node.  A
    # missing marker makes its semantic name unresolved, not its electrostatic
    # influence irrelevant.  This is required by the one-marker star/concentric
    # sweeps, which still solve two signal conductors in Q3D.
    queryable = np.ones(len(nodes), dtype=bool)
    marker_counts = [item[3] for item in terminal_rows] + [0] * len(floating_rows)
    return nodes, node_ids, node_roles, queryable, marker_counts


def _self_spectrum(shape: Any, scale: float) -> np.ndarray:
    points, weights = _boundary_quadrature(shape, 96)
    if not len(points):
        return np.zeros(16, dtype=np.float64)
    center = np.asarray(shape.centroid.coords[0], dtype=np.float64)
    radius = np.sqrt(np.sum((points - center) ** 2, axis=1)) / max(scale, 1e-12)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.log1p(soft_histogram(radius, weights, V4_NORMALIZED_SELF_EDGES))


def _ground_spectrum(shape: Any, ground: Any | None, scale: float, edges: np.ndarray, normalized: bool) -> np.ndarray:
    shapely = _require_shapely()
    points, weights = _boundary_quadrature(shape, 96)
    if not len(points) or ground is None:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    distances = shapely.distance(shapely.points(points), ground)
    if normalized:
        distances = distances / max(scale, 1e-12)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.log1p(soft_histogram(distances, weights, edges))


def _edge_features(first: Any, second: Any, global_scale: float) -> np.ndarray:
    pair_scale = math.sqrt(max(float(first.area + second.area), 1e-12))
    first_points, first_weights = _boundary_quadrature(first, 64)
    second_points, second_weights = _boundary_quadrature(second, 64)
    distances = _pair_distances(first_points, second_points)
    pair_weights = np.outer(first_weights, second_weights)
    normalized_weights = pair_weights / max(float(pair_weights.sum()), 1e-12)
    flattened = distances.reshape(-1)
    flattened_weights = normalized_weights.reshape(-1)
    median = _weighted_quantile(flattened, flattened_weights, 0.5)
    p90 = _weighted_quantile(flattened, flattened_weights, 0.9)
    gap = float(first.distance(second))
    centers = np.asarray(first.centroid.coords[0]) - np.asarray(second.centroid.coords[0])
    center_distance = float(np.linalg.norm(centers))
    area_total = float(first.area + second.area)
    perimeter_total = float(first.length + second.length)
    union = first.union(second)
    components, holes = _component_and_hole_count(union)
    metrics = np.asarray(
        [
            math.log1p(pair_scale),
            pair_scale / max(global_scale, 1e-12),
            math.log1p(gap),
            gap / max(pair_scale, 1e-12),
            math.log1p(median),
            median / max(pair_scale, 1e-12),
            math.log1p(p90),
            p90 / max(pair_scale, 1e-12),
            math.log1p(center_distance),
            center_distance / max(pair_scale, 1e-12),
            min(float(first.area), float(second.area)) / max(max(float(first.area), float(second.area)), 1e-12),
            min(float(first.length), float(second.length)) / max(max(float(first.length), float(second.length)), 1e-12),
            perimeter_total / math.sqrt(max(area_total, 1e-12)),
            area_total / max(float(union.convex_hull.area), 1e-12),
            float(components),
            float(holes),
        ]
    )
    normalized_hist = np.log1p(
        soft_histogram(flattened / max(pair_scale, 1e-12), flattened_weights, V4_NORMALIZED_PAIR_EDGES)
    )
    absolute_hist = np.log1p(soft_histogram(flattened, flattened_weights, V4_ABSOLUTE_PAIR_EDGES_UM))
    operator_weights = np.outer(first_weights / pair_scale, second_weights / pair_scale)
    operator = _operator_values(distances / max(pair_scale, 1e-12), operator_weights)
    topology = np.zeros(16, dtype=np.float64)
    for index, radius in enumerate(V4_TOPOLOGY_RADII_NORMALIZED):
        buffered = union.buffer(float(radius * pair_scale))
        count, hole_count = _component_and_hole_count(buffered)
        topology[index] = math.log1p(count)
        topology[8 + index] = math.log1p(hole_count)
    vector = np.concatenate([metrics, normalized_hist, absolute_hist, operator, topology])
    if vector.shape != (V4_EDGE_DIMENSIONS,):
        raise RuntimeError(f"v4 edge feature size {vector.shape} != ({V4_EDGE_DIMENSIONS},)")
    return vector


def _bem_proxy(nodes: Sequence[Any], ground_lines: Sequence[Any], scale: float) -> tuple[np.ndarray, dict[str, float]]:
    roles: list[Any | Sequence[Any]] = list(nodes)
    if ground_lines:
        roles.append(ground_lines)
    points = []
    lengths = []
    labels = []
    per_role = max(12, min(32, 320 // max(len(roles), 1)))
    for index, role in enumerate(roles):
        role_points, role_weights = _boundary_quadrature(role, per_role)
        if not len(role_points):
            continue
        points.append(role_points)
        lengths.append(role_weights)
        labels.append(np.full(len(role_points), index, dtype=int))
    role_count = len(roles)
    if len(points) < 2:
        return np.zeros((role_count, role_count)), {
            "available": 0.0,
            "condition_number": 0.0,
            "sample_count": float(sum(len(item) for item in points)),
            "projection_residual": 0.0,
        }
    coordinates = np.vstack(points)
    segment = np.concatenate(lengths)
    owners = np.concatenate(labels)
    distances = _pair_distances(coordinates, coordinates) / max(scale, 1e-12)
    nonzero = distances[distances > 1e-12]
    reference = 10.0 * max(float(np.max(nonzero)) if len(nonzero) else 1.0, 1.0)
    np.fill_diagonal(distances, 1.0)
    epsilon = VACUUM_PERMITTIVITY
    green = np.log(reference / np.maximum(distances, 1e-12)) / (2.0 * np.pi * epsilon)
    normalized_segment = np.maximum(segment / max(scale, 1e-12), 1e-12)
    np.fill_diagonal(green, (np.log(2.0 * reference / normalized_segment) + 1.0) / (2.0 * np.pi * epsilon))
    selector = np.stack([(owners == index).astype(float) for index in range(role_count)], axis=1)
    # Symmetric conductor configurations can make the collocation system nearly
    # singular, causing nanometre-scale GDS quantization to dominate the raw
    # inverse.  A fixed-rank truncated-SVD solve (eight modes per physical
    # conductor/ground role) is a deterministic regularization of this
    # low-fidelity feature, not a parameter fitted to simulation labels.
    left_vectors, singular_values, right_vectors = np.linalg.svd(green, full_matrices=False)
    retained_rank = min(len(singular_values), max(8, 8 * role_count))
    retained = np.arange(len(singular_values)) < retained_rank
    inverse = (right_vectors[retained].T / singular_values[retained]) @ left_vectors[:, retained].T
    charges = inverse @ selector
    condition = float(singular_values[0] / singular_values[retained][-1])
    raw = selector.T @ charges
    raw = 0.5 * (raw + raw.T)
    projected = np.zeros_like(raw)
    for first in range(role_count):
        for second in range(first + 1, role_count):
            coupling = abs(float(raw[first, second]))
            projected[first, first] += coupling
            projected[second, second] += coupling
            projected[first, second] = projected[second, first] = -coupling
    residual = float(np.linalg.norm(projected - raw) / max(np.linalg.norm(raw), 1e-30))
    scaled_fF = projected * scale * 1e-6 * 1e15
    return scaled_fF, {
        "available": 1.0,
        "condition_number": condition,
            "sample_count": float(len(coordinates)),
            "projection_residual": residual,
            "retained_rank": float(retained_rank),
        }


def _node_features(
    nodes: Sequence[Any],
    roles: Sequence[str],
    marker_counts: Sequence[int],
    ground: Any | None,
    proxy: np.ndarray,
    ground_index: int | None,
) -> np.ndarray:
    features = np.zeros((len(nodes), V4_NODE_DIMENSIONS), dtype=np.float64)
    for index, (shape, role, marker_count) in enumerate(zip(nodes, roles, marker_counts)):
        scale = math.sqrt(max(float(shape.area), 1e-12))
        components, holes = _component_and_hole_count(shape)
        eccentricity, minor, major = _covariance_invariants(shape, scale)
        ground_gap = float(shape.distance(ground)) if ground is not None else 0.0
        proxy_ground = abs(float(proxy[index, ground_index])) if ground_index is not None else 0.0
        role_values = np.asarray(
            [
                float(role == "terminal"),
                float(role in {"floating", "unmarked"}),
                math.log1p(marker_count),
                float(components),
                float(holes),
                float(ground is not None),
                1.0,
                0.0,
            ]
        )
        hull_area = max(float(shape.convex_hull.area), 1e-12)
        morphology = np.asarray(
            [
                math.log1p(scale),
                math.log1p(float(shape.area)),
                math.log1p(float(shape.length)),
                float(shape.length) / max(scale, 1e-12),
                float(shape.area) / hull_area,
                eccentricity,
                minor,
                major,
                math.log1p(ground_gap),
                ground_gap / max(scale, 1e-12),
                math.log1p(proxy_ground),
                math.log1p(float(shape.convex_hull.length)),
                float(components),
                float(holes),
                0.0,
                0.0,
            ]
        )
        features[index] = np.concatenate(
            [
                role_values,
                morphology,
                _self_spectrum(shape, scale),
                _ground_spectrum(shape, ground, scale, V4_NORMALIZED_GROUND_EDGES, True),
                _ground_spectrum(shape, ground, scale, V4_ABSOLUTE_GROUND_EDGES_UM, False),
            ]
        )
    return features


def _global_features(
    nodes: Sequence[Any],
    roles: Sequence[str],
    marker_counts: Sequence[int],
    ground: Any | None,
    ground_lines: Sequence[Any],
    proxy: np.ndarray,
    proxy_quality: Mapping[str, float],
) -> np.ndarray:
    shapely = _require_shapely()
    union = shapely.union_all(nodes)
    scale = math.sqrt(max(float(union.area), 1e-12))
    components, holes = _component_and_hole_count(union)
    pair_gaps = [
        float(nodes[first].distance(nodes[second]))
        for first in range(len(nodes))
        for second in range(first + 1, len(nodes))
    ]
    minimum_pair_gap = min(pair_gaps) if pair_gaps else 0.0
    counts_extent = np.asarray(
        [
            float(sum(role == "terminal" for role in roles)),
            float(sum(role in {"floating", "unmarked"} for role in roles)),
            float(len(nodes)),
            float(sum(marker_counts)),
            math.log1p(float(union.area)),
            math.log1p(float(union.length)),
            math.log1p(scale),
            float(union.length) / max(scale, 1e-12),
            float(union.area) / max(float(union.convex_hull.area), 1e-12),
            float(components),
            float(holes),
            math.log1p(float(union.convex_hull.length)),
            math.log1p(minimum_pair_gap),
            minimum_pair_gap / max(scale, 1e-12),
            float(bool(ground_lines)),
            math.log1p(len(nodes) * max(len(nodes) - 1, 0) / 2),
        ]
    )
    topology = np.zeros(16, dtype=np.float64)
    for index, radius in enumerate(V4_TOPOLOGY_RADII_NORMALIZED):
        count, hole_count = _component_and_hole_count(union.buffer(float(radius * scale)))
        topology[index] = math.log1p(count)
        topology[8 + index] = math.log1p(hole_count)
    gaps = np.asarray([float(node.distance(ground)) for node in nodes]) if ground is not None else np.zeros(len(nodes))
    ground_components, ground_holes = _component_and_hole_count(ground) if ground is not None else (0, 0)
    local_length = float(sum(line.length for line in ground_lines))
    ground_values = np.asarray(
        [
            float(bool(ground_lines)),
            float(len(ground_lines)),
            math.log1p(local_length),
            local_length / max(scale, 1e-12),
            float(ground_components),
            float(ground_holes),
            math.log1p(float(np.min(gaps))) if len(gaps) else 0.0,
            math.log1p(float(np.mean(gaps))) if len(gaps) else 0.0,
            math.log1p(float(np.max(gaps))) if len(gaps) else 0.0,
            float(np.min(gaps) / max(scale, 1e-12)) if len(gaps) else 0.0,
            float(np.mean(gaps) / max(scale, 1e-12)) if len(gaps) else 0.0,
            float(np.max(gaps) / max(scale, 1e-12)) if len(gaps) else 0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    eigenvalues = np.linalg.eigvalsh(proxy) if proxy.size else np.zeros(0)
    quantiles = np.quantile(eigenvalues, np.linspace(0.0, 1.0, 8)) if len(eigenvalues) else np.zeros(8)
    operator_values = np.asarray(
        [
            float(proxy_quality["available"]),
            float(proxy.shape[0]),
            math.log1p(proxy_quality["sample_count"]),
            math.log1p(max(proxy_quality["condition_number"], 0.0)),
            float(proxy_quality["projection_residual"]),
            math.log1p(abs(float(np.trace(proxy)))) if proxy.size else 0.0,
            math.log1p(float(np.linalg.norm(proxy))) if proxy.size else 0.0,
            math.log1p(float(np.sum(np.abs(np.triu(proxy, 1))))) if proxy.size else 0.0,
            *np.sign(quantiles) * np.log1p(np.abs(quantiles)),
        ]
    )
    return np.concatenate([counts_extent, topology, ground_values, operator_values])


def encode_v4_graph(
    gds_path: str | Path,
    *,
    layer_contract: str | Path | Mapping[str, Any] | None = None,
) -> PlanarGraphV4:
    """Encode one standardized planar GDS file into an arbitrary-size graph."""
    shapely = _require_shapely()
    geometry = read_layer_geometry(gds_path)
    contract = _load_contract(geometry, layer_contract)
    role_keys, entries = _contract_roles(contract)
    conductor = _union_layers(geometry, role_keys["conductor"])
    if conductor is None:
        raise ValueError("No conductor geometry exists on the declared v4 layers.")
    ground = _union_layers(geometry, role_keys["ground"])
    etch = _union_layers(geometry, role_keys["etch"])
    if ground is not None and etch is not None:
        ground = ground.difference(etch)
    nodes, node_ids, node_roles, queryable, marker_counts = _terminal_nodes(
        conductor, geometry, role_keys["port"], entries
    )

    # Re-origin before every calculation to make large-coordinate translations
    # bitwise stable under finite-precision geometric operations.
    origin = np.asarray(conductor.bounds[:2], dtype=np.float64)
    def move(shape: Any) -> Any:
        return shapely.transform(shape, lambda points: points - origin)

    nodes = [move(node) for node in nodes]
    conductor = move(conductor)
    ground = move(ground) if ground is not None else None
    ground_lines = _local_ground_lines(ground)
    scale = math.sqrt(max(float(conductor.area), 1e-12))
    proxy, proxy_quality = _bem_proxy(nodes, ground_lines, scale)
    ground_index = len(nodes) if ground_lines else None
    node_features = _node_features(nodes, node_roles, marker_counts, ground, proxy, ground_index)
    edge_features = np.zeros((len(nodes), len(nodes), V4_EDGE_DIMENSIONS), dtype=np.float64)
    for first in range(len(nodes)):
        for second in range(first + 1, len(nodes)):
            edge = _edge_features(nodes[first], nodes[second], scale)
            edge_features[first, second] = edge
            edge_features[second, first] = edge
    global_features = _global_features(
        nodes, node_roles, marker_counts, ground, ground_lines, proxy, proxy_quality
    )
    metadata = {
        "model": PLANAR_TERMINAL_GRAPH_V4_MODEL,
        "schema_version": PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION,
        "gds_path": str(Path(gds_path)),
        "contract_schema_version": contract.get("schema_version"),
        "terminal_count": int(queryable.sum()),
        "unmarked_conductor_count": int(sum(role == "unmarked" for role in node_roles)),
        "node_count": len(nodes),
        "ground_available": bool(ground_lines),
        "outer_ground_crop_used": False,
        "proxy_condition_number": proxy_quality["condition_number"],
        "proxy_projection_residual": proxy_quality["projection_residual"],
    }
    return PlanarGraphV4(
        node_features=np.nan_to_num(node_features).astype(np.float32),
        edge_features=np.nan_to_num(edge_features).astype(np.float32),
        global_features=np.nan_to_num(global_features).astype(np.float32),
        node_ids=tuple(node_ids),
        node_roles=tuple(node_roles),
        queryable_mask=queryable,
        proxy_matrix_fF=np.nan_to_num(proxy),
        metadata=metadata,
    )


def layout_view_v4(graph: PlanarGraphV4) -> np.ndarray:
    """Return a permutation-invariant fixed vector for maps and retrieval."""
    node = graph.node_features.astype(np.float64)
    node_pool = np.concatenate([node.mean(axis=0), node.std(axis=0), node.max(axis=0)])
    indices = np.triu_indices(len(node), 1)
    edges = graph.edge_features[indices].astype(np.float64)
    if len(edges):
        edge_pool = np.concatenate([edges.mean(axis=0), edges.std(axis=0), edges.max(axis=0)])
    else:
        edge_pool = np.zeros(3 * V4_EDGE_DIMENSIONS, dtype=np.float64)
    vector = np.concatenate([graph.global_features, node_pool, edge_pool]).astype(np.float32)
    if vector.shape != (V4_LAYOUT_VIEW_DIMENSIONS,):
        raise RuntimeError(f"v4 layout view has shape {vector.shape}, expected ({V4_LAYOUT_VIEW_DIMENSIONS},)")
    return vector


def pair_view_v4(graph: PlanarGraphV4, first: int | str, second: int | str) -> np.ndarray:
    """Return a reciprocal pair-query vector retaining every other conductor."""
    if isinstance(first, str):
        first = graph.node_ids.index(first)
    if isinstance(second, str):
        second = graph.node_ids.index(second)
    if first == second or min(first, second) < 0 or max(first, second) >= len(graph.node_ids):
        raise ValueError(f"Invalid v4 pair query ({first}, {second}) for {len(graph.node_ids)} nodes.")
    if not graph.queryable_mask[first] or not graph.queryable_mask[second]:
        raise ValueError("Pair queries require two conductor nodes retained by the graph contract.")
    endpoint_sum = graph.node_features[first] + graph.node_features[second]
    endpoint_difference = np.abs(graph.node_features[first] - graph.node_features[second])
    selected_edge = graph.edge_features[first, second]
    messages = []
    for other in range(len(graph.node_ids)):
        if other in (first, second):
            continue
        left = graph.edge_features[first, other, :32]
        right = graph.edge_features[second, other, :32]
        messages.append(np.concatenate([left + right, np.abs(left - right)]))
    if messages:
        message = np.asarray(messages, dtype=np.float64)
        context = np.concatenate([message.mean(axis=0), message.std(axis=0), message.max(axis=0)])
    else:
        context = np.zeros(192, dtype=np.float64)
    vector = np.concatenate(
        [graph.global_features, endpoint_sum, endpoint_difference, selected_edge, context]
    ).astype(np.float32)
    if vector.shape != (V4_PAIR_VIEW_DIMENSIONS,):
        raise RuntimeError(f"v4 pair view has shape {vector.shape}, expected ({V4_PAIR_VIEW_DIMENSIONS},)")
    return vector


def encode_v4_pair(
    gds_path: str | Path,
    first: int | str = 0,
    second: int | str = 1,
    *,
    layer_contract: str | Path | Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    graph: PlanarGraphV4 | None = None,
) -> np.ndarray:
    """Return the production planar pair view: v3 pair physics + v4 context.

    The first 240 coordinates deliberately preserve the validated invariant v3
    fixed-stack pair core.  The final 192 coordinates pool symmetric messages
    from every other conductor in the arbitrary-size v4 graph.  Two-terminal
    layouts therefore retain the exact established pair representation, while a
    third or sixteenth conductor cannot be silently ignored.
    """
    graph = graph or encode_v4_graph(gds_path, layer_contract=layer_contract)
    if isinstance(first, str):
        first = graph.node_ids.index(first)
    if isinstance(second, str):
        second = graph.node_ids.index(second)
    if first == second or min(first, second) < 0 or max(first, second) >= len(graph.node_ids):
        raise ValueError(f"Invalid v4 pair query ({first}, {second}) for {len(graph.node_ids)} nodes.")

    # Canonicalize reciprocal endpoints by descending physical conductor size,
    # matching the long-standing geometry-only fallback used by v2/v3.  The
    # remaining invariant node features break exact size ties; names and port
    # layer indices never enter the numerical ordering.
    ordered = sorted(
        (int(first), int(second)),
        key=lambda index: (
            -round(float(graph.node_features[index, 9]), 7),
            -round(float(graph.node_features[index, 10]), 7),
            tuple(np.asarray(graph.node_features[index], dtype=np.float64).round(7)),
        ),
    )
    geometry = read_layer_geometry(gds_path)
    contract = _load_contract(geometry, layer_contract)
    role_keys, _ = _contract_roles(contract)
    roles = {
        **{key: "conductor" for key in role_keys["conductor"]},
        **{key: "domain" for key in role_keys["ground"]},
        **{key: "etch" for key in role_keys["etch"]},
        **{key: "port" for key in role_keys["port"]},
    }
    v3 = encode_v3(
        gds_path,
        environment=environment,
        terminal_pair=(ordered[0], ordered[1]),
        layer_roles=roles,
    )
    graph_pair = pair_view_v4(graph, first, second)
    context = graph_pair[288:]
    vector = np.concatenate([v3[V3_BLOCK_SLICES["fixed_stack_core"]], context]).astype(np.float32)
    if vector.shape != (V4_HYBRID_PAIR_DIMENSIONS,):
        raise RuntimeError(
            f"v4 hybrid pair view has shape {vector.shape}, expected ({V4_HYBRID_PAIR_DIMENSIONS},)"
        )
    return vector


def planar_terminal_graph_v4_schema() -> dict[str, Any]:
    """Return the self-describing v4 graph and derived-view contract."""
    return {
        "model": PLANAR_TERMINAL_GRAPH_V4_MODEL,
        "embedding_schema_version": PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION,
        "experimental": True,
        "fitted_on_catalogue": False,
        "simulation_results_used": False,
        "implemented_geometry": "single-plane planar",
        "canonical_representation": {
            "node_features": ["terminal_count", V4_NODE_DIMENSIONS],
            "edge_features": ["terminal_count", "terminal_count", V4_EDGE_DIMENSIONS],
            "global_features": [V4_GLOBAL_DIMENSIONS],
            "terminal_count_cap": None,
        },
        "derived_views": {
            "layout": {"dimensions": V4_LAYOUT_VIEW_DIMENSIONS, "permutation_invariant": True},
            "pair": {
                "dimensions": V4_PAIR_VIEW_DIMENSIONS,
                "endpoint_swap_invariant": True,
                "other_terminal_permutation_invariant": True,
            },
            "hybrid_pair": {
                "dimensions": V4_HYBRID_PAIR_DIMENSIONS,
                "blocks": {"v3_fixed_stack_pair_core": 240, "all_other_conductor_context": 192},
                "recommended_for": "ordinary regressors and planar capacitance transfer",
            },
        },
        "symmetry_contract": {
            "translation": "invariant",
            "in_plane_rotation": "invariant for isotropic planar materials",
            "reflection": "invariant for parity-symmetric planar materials",
            "terminal_relabeling": "graph equivariant; layout and corresponding pair views invariant",
            "scale": "normalized shape and absolute physical measurements are both retained",
        },
        "future_layer_stack_contract": {
            "format": "versioned JSON or XML mapped into a common internal representation",
            "required_fields": ["GDS layer/datatype", "plane z", "material", "thickness", "terminal net"],
            "current_behavior": "reject multiple physical conductor planes rather than emit misleading features",
        },
        "limitations": [
            "The pooled layout view is diagnostic and is not mathematically injective over all possible layouts.",
            "The boundary operator is a deterministic two-dimensional low-fidelity feature, not simulation truth.",
            "GDS alone cannot determine terminal nets, floating status, materials, or physical layer heights.",
        ],
    }


__all__ = [
    "PLANAR_TERMINAL_GRAPH_V4_MODEL",
    "PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION",
    "PlanarGraphV4",
    "V4_EDGE_DIMENSIONS",
    "V4_GLOBAL_DIMENSIONS",
    "V4_HYBRID_PAIR_DIMENSIONS",
    "V4_LAYOUT_VIEW_DIMENSIONS",
    "V4_NODE_DIMENSIONS",
    "V4_PAIR_VIEW_DIMENSIONS",
    "encode_v4_graph",
    "encode_v4_pair",
    "layout_view_v4",
    "pair_view_v4",
    "planar_terminal_graph_v4_schema",
]
