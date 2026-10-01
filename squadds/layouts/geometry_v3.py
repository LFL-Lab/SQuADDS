"""Rigid-motion-invariant, scale-aware electrostatic layout signatures.

``capacitance-operator-v3`` is an additive successor to
``universal-geometry-v2``.  It does not alter any v0--v2 artifact or API.  The
representation is query-conditioned: a vector describes one ordered terminal
pair in one layout and swapping terminal labels permutes the terminal-specific
channels rather than changing the physical pair interaction.

The scientific contract is deliberately narrower and stronger than v2's:

* geometry channels depend on distances, measures, covariance eigenvalues, or
  vector buffering, so laboratory-frame translation, rotation, and reflection
  are absent in the continuum;
* normalized shape and absolute physical scale are stored separately;
* the arbitrary outer simulation-domain edge is excluded from ground features;
* a local grounded boundary-element solve supplies a deterministic low-fidelity
  Maxwell-capacitance proxy; and
* substrate and metal-stack quantities are explicit inputs rather than hidden
  family offsets.

The boundary-element block is a feature, not a replacement for a converged 3-D
field solve.  It uses a dimensionless two-dimensional logarithmic Green kernel
and lifts the resulting per-depth response by the characteristic device scale.
Its purpose is to provide a common physical mean function for a supervised
high-fidelity correction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .geometry_v2 import (
    DEFAULT_LAYER_ROLES,
    VACUUM_PERMITTIVITY,
    _role_geometry,
    _terminals,
    read_layer_geometry,
    soft_histogram,
)

CAPACITANCE_OPERATOR_V3_MODEL = "capacitance-operator-v3"
CAPACITANCE_OPERATOR_V3_SCHEMA_VERSION = "3.0.0-experimental"

V3_METRIC_BLOCK_SIZE = 40
V3_INTERACTION_BINS = 32
V3_INTERACTION_BLOCK_SIZE = 3 * V3_INTERACTION_BINS
V3_OPERATOR_VALUES_PER_PAIR = 16
V3_OPERATOR_BLOCK_SIZE = 3 * V3_OPERATOR_VALUES_PER_PAIR
V3_TOPOLOGY_RADII = 16
V3_TOPOLOGY_BLOCK_SIZE = 2 * V3_TOPOLOGY_RADII
V3_ELECTROSTATIC_BLOCK_SIZE = 24
V3_ENVIRONMENT_BLOCK_SIZE = 16
V3_DIMENSIONS = (
    V3_METRIC_BLOCK_SIZE
    + V3_INTERACTION_BLOCK_SIZE
    + V3_OPERATOR_BLOCK_SIZE
    + V3_TOPOLOGY_BLOCK_SIZE
    + V3_ELECTROSTATIC_BLOCK_SIZE
    + V3_ENVIRONMENT_BLOCK_SIZE
)

# The explicit stack channels are intentionally separable.  On a fixed-stack
# catalogue they should normally be omitted from a learned metric: thickness /
# device-size ratios otherwise become accidental component-family identifiers.
# The electrostatic block in the core still uses the supplied permittivity.
V3_FIXED_STACK_DIMENSIONS = V3_DIMENSIONS - V3_ENVIRONMENT_BLOCK_SIZE
V3_BLOCK_SLICES = {
    "metrics": slice(0, V3_METRIC_BLOCK_SIZE),
    "interactions": slice(V3_METRIC_BLOCK_SIZE, V3_METRIC_BLOCK_SIZE + V3_INTERACTION_BLOCK_SIZE),
    "operator": slice(
        V3_METRIC_BLOCK_SIZE + V3_INTERACTION_BLOCK_SIZE,
        V3_METRIC_BLOCK_SIZE + V3_INTERACTION_BLOCK_SIZE + V3_OPERATOR_BLOCK_SIZE,
    ),
    "topology": slice(
        V3_METRIC_BLOCK_SIZE + V3_INTERACTION_BLOCK_SIZE + V3_OPERATOR_BLOCK_SIZE,
        V3_METRIC_BLOCK_SIZE
        + V3_INTERACTION_BLOCK_SIZE
        + V3_OPERATOR_BLOCK_SIZE
        + V3_TOPOLOGY_BLOCK_SIZE,
    ),
    "electrostatic": slice(
        V3_FIXED_STACK_DIMENSIONS - V3_ELECTROSTATIC_BLOCK_SIZE,
        V3_FIXED_STACK_DIMENSIONS,
    ),
    "environment": slice(V3_FIXED_STACK_DIMENSIONS, V3_DIMENSIONS),
    "fixed_stack_core": slice(0, V3_FIXED_STACK_DIMENSIONS),
    "complete": slice(0, V3_DIMENSIONS),
}

if V3_DIMENSIONS != 256:  # pragma: no cover - schema arithmetic guard
    raise RuntimeError(f"capacitance-operator-v3 must have 256 dimensions, received {V3_DIMENSIONS}.")

V3_NORMALIZED_DISTANCE_EDGES = np.logspace(-3.0, 2.0, V3_INTERACTION_BINS + 1)
V3_TOPOLOGY_RADII_NORMALIZED = np.logspace(-3.0, 0.0, V3_TOPOLOGY_RADII)

INTERACTION_SAMPLES = 64
BEM_SAMPLES_PER_ROLE = 28

V3_METRIC_NAMES = [
    "log1p_characteristic_scale_um",
    "log1p_total_terminal_area_um2",
    "log1p_total_terminal_perimeter_um",
    "terminal_union_compactness",
    "terminal_union_hull_fill",
    "terminal_count",
    "log1p_terminal_0_area_um2",
    "log1p_terminal_1_area_um2",
    "log1p_terminal_0_perimeter_um",
    "log1p_terminal_1_perimeter_um",
    "terminal_0_area_fraction",
    "terminal_1_area_fraction",
    "terminal_0_perimeter_fraction",
    "terminal_1_perimeter_fraction",
    "terminal_0_compactness",
    "terminal_1_compactness",
    "log1p_terminal_centroid_separation_um",
    "terminal_centroid_separation_per_scale",
    "log1p_minimum_terminal_gap_um",
    "minimum_terminal_gap_per_scale",
    "log1p_median_boundary_distance_um",
    "median_boundary_distance_per_scale",
    "log1p_p90_boundary_distance_um",
    "p90_boundary_distance_per_scale",
    "log1p_terminal_0_ground_gap_um",
    "log1p_terminal_1_ground_gap_um",
    "terminal_0_ground_gap_per_scale",
    "terminal_1_ground_gap_per_scale",
    "terminal_0_boundary_eccentricity",
    "terminal_1_boundary_eccentricity",
    "terminal_0_radial_dispersion_per_scale",
    "terminal_1_radial_dispersion_per_scale",
    "terminal_union_component_count",
    "terminal_union_hole_count",
    "log1p_local_ground_boundary_length_um",
    "local_ground_boundary_length_per_scale",
    "ground_available",
    "absolute_terminal_area_asymmetry",
    "absolute_terminal_perimeter_asymmetry",
    "terminal_area_balance",
]

V3_ELECTROSTATIC_NAMES = [
    *[f"signed_log1p_scaled_proxy_C_{i}{j}_fF" for i in range(3) for j in range(i, 3)],
    *[f"signed_log1p_dimensionless_proxy_C_{i}{j}" for i in range(3) for j in range(i, 3)],
    "signed_log1p_scaled_proxy_eigenvalue_0_fF",
    "signed_log1p_scaled_proxy_eigenvalue_1_fF",
    "signed_log1p_scaled_proxy_eigenvalue_2_fF",
    "signed_log1p_scaled_proxy_trace_fF",
    "log1p_scaled_proxy_mutual_01_fF",
    "log1p_scaled_proxy_terminal_0_ground_fF",
    "log1p_scaled_proxy_terminal_1_ground_fF",
    "log1p_green_condition_number",
    "electrostatic_role_count",
    "log1p_electrostatic_sample_count",
    "log1p_electrostatic_reference_scale_um",
    "electrostatic_proxy_available",
]

V3_ENVIRONMENT_NAMES = [
    "relative_permittivity",
    "effective_relative_permittivity",
    "log1p_substrate_thickness_um",
    "substrate_thickness_per_scale",
    "log1p_metal_thickness_um",
    "metal_thickness_per_scale",
    "log1p_vacuum_height_um",
    "vacuum_height_per_scale",
    "in_plane_permittivity_x",
    "in_plane_permittivity_y",
    "out_of_plane_permittivity",
    "in_plane_isotropic",
    "stack_available",
    "ground_reference_available",
    "reserved_environment_0",
    "reserved_environment_1",
]


def _require_shapely():
    try:
        import shapely
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("capacitance-operator-v3 requires shapely: uv sync --extra gds") from exc
    return shapely


def _signed_log1p(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.log1p(np.abs(array))


def _polygon_parts(shape: Any) -> list[Any]:
    return [part for part in getattr(shape, "geoms", [shape]) if part.geom_type == "Polygon" and part.area > 0]


def _component_and_hole_count(shape: Any) -> tuple[int, int]:
    parts = _polygon_parts(shape)
    return len(parts), sum(len(part.interiors) for part in parts)


def _ground_geometry(grouped: Mapping[str, list[tuple[tuple[int, int], Any]]]) -> Any | None:
    """Return physical ground metal and interpret a legacy etch subtractively."""
    shapely = _require_shapely()
    domains = [shape for _, shape in grouped["domain"]]
    if not domains:
        return None
    ground = shapely.union_all(domains)
    etches = [shape for _, shape in grouped["etch"]]
    if etches:
        ground = ground.difference(shapely.union_all(etches))
    return None if ground.is_empty else ground


def _local_ground_lines(ground: Any | None, conductor: Any) -> list[Any]:
    """Select moat-facing ground rings and ignore the arbitrary outer crop edge."""
    if ground is None:
        return []
    shapely = _require_shapely()
    interior_lines = []
    exterior_lines = []
    for polygon in _polygon_parts(ground):
        exterior_lines.append(shapely.LineString(polygon.exterior.coords))
        interior_lines.extend(shapely.LineString(ring.coords) for ring in polygon.interiors)
    if interior_lines:
        return interior_lines
    if not exterior_lines:
        return []
    distances = np.asarray([line.distance(conductor) for line in exterior_lines])
    closest = float(np.min(distances))
    tolerance = max(1e-7, 1e-6 * max(math.sqrt(max(float(conductor.area), 1e-12)), 1.0))
    return [line for line, distance in zip(exterior_lines, distances) if distance <= closest + tolerance]


def _shape_lines(shape: Any) -> list[Any]:
    shapely = _require_shapely()
    if shape is None:
        return []
    if shape.geom_type in {"LineString", "LinearRing"}:
        return [shapely.LineString(shape.coords)]
    if shape.geom_type == "MultiLineString":
        return list(shape.geoms)
    lines = []
    for polygon in _polygon_parts(shape):
        lines.append(shapely.LineString(polygon.exterior.coords))
        lines.extend(shapely.LineString(ring.coords) for ring in polygon.interiors)
    return lines


def _segment_quadrature(shape_or_lines: Any | Sequence[Any], target: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample every polygon segment as an unordered quadrature set.

    Segment-local midpoint rules avoid dependence on a polygon ring's arbitrary
    starting vertex.  A rigid transformation therefore moves the sample set as a
    whole without changing any downstream distance aggregation.
    """
    if isinstance(shape_or_lines, Sequence) and not hasattr(shape_or_lines, "geom_type"):
        lines = list(shape_or_lines)
    else:
        lines = _shape_lines(shape_or_lines)
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    for line in lines:
        coordinates = np.asarray(line.coords, dtype=np.float64)
        for start, stop in zip(coordinates[:-1], coordinates[1:]):
            length = float(np.linalg.norm(stop - start))
            if length > 1e-12:
                segments.append((start, stop, length))
    if not segments:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    total = sum(item[2] for item in segments)
    points = []
    weights = []
    for start, stop, length in segments:
        count = max(1, int(round(target * length / total)))
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        points.append(start[None, :] + fractions[:, None] * (stop - start)[None, :])
        weights.append(np.full(count, length / count, dtype=np.float64))
    return np.vstack(points), np.concatenate(weights)


def _pair_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    delta = first[:, None, :] - second[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    flattened_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(flattened) == 0 or float(flattened_weights.sum()) <= 0:
        return 0.0
    order = np.argsort(flattened)
    ordered = flattened[order]
    cumulative = np.cumsum(flattened_weights[order])
    target = probability * cumulative[-1]
    return float(ordered[min(int(np.searchsorted(cumulative, target, side="left")), len(ordered) - 1)])


def _boundary_invariants(points: np.ndarray, weights: np.ndarray, scale: float) -> tuple[float, float]:
    if len(points) == 0 or float(weights.sum()) <= 0:
        return 0.0, 0.0
    normalized_weights = weights / weights.sum()
    center = np.sum(points * normalized_weights[:, None], axis=0)
    centered = points - center
    covariance = (centered * normalized_weights[:, None]).T @ centered
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    eccentricity = math.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[-1], 1e-12)))
    dispersion = math.sqrt(max(float(np.trace(covariance)), 0.0)) / max(scale, 1e-12)
    return eccentricity, dispersion


def _interaction_spectrum(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_points, first_weights = first
    second_points, second_weights = second
    if len(first_points) == 0 or len(second_points) == 0:
        return np.zeros(V3_INTERACTION_BINS), np.zeros((0, 0)), np.zeros((0, 0))
    distances = _pair_distances(first_points, second_points) / max(scale, 1e-12)
    pair_weights = np.outer(first_weights / scale, second_weights / scale)
    histogram = soft_histogram(
        distances.reshape(-1),
        pair_weights.reshape(-1),
        V3_NORMALIZED_DISTANCE_EDGES,
    )
    return np.log1p(histogram), distances, pair_weights


def _operator_values(distances: np.ndarray, pair_weights: np.ndarray) -> np.ndarray:
    if distances.size == 0:
        return np.zeros(V3_OPERATOR_VALUES_PER_PAIR, dtype=np.float64)
    safe = np.maximum(distances, 1e-6)
    weight = np.sqrt(np.maximum(pair_weights, 0.0))
    log_kernel = np.abs(-np.log(safe)) * weight
    inverse_kernel = weight / np.sqrt(safe * safe + 1e-4)
    answer = []
    for kernel in (log_kernel, inverse_kernel):
        singular = np.linalg.svd(kernel, compute_uv=False)
        padded = np.pad(singular[:8], (0, max(0, 8 - len(singular))))[:8]
        answer.extend(np.log1p(np.maximum(padded, 0.0)))
    return np.asarray(answer, dtype=np.float64)


def _environment_values(environment: Mapping[str, Any] | None, scale: float, ground_available: bool) -> tuple[np.ndarray, dict[str, float]]:
    supplied = dict(environment or {})
    epsilon_r = float(supplied.get("relative_permittivity", supplied.get("epsilon_r", 1.0)))
    epsilon_eff = float(supplied.get("effective_relative_permittivity", 0.5 * (1.0 + epsilon_r)))
    substrate = float(supplied.get("substrate_thickness_um", 0.0))
    metal = float(supplied.get("metal_thickness_um", 0.0))
    vacuum = float(supplied.get("vacuum_height_um", supplied.get("vacuum_box_size_z_um", 0.0)))
    epsilon_x = float(supplied.get("in_plane_permittivity_x", epsilon_r))
    epsilon_y = float(supplied.get("in_plane_permittivity_y", epsilon_r))
    epsilon_z = float(supplied.get("out_of_plane_permittivity", epsilon_r))
    tolerance = 1e-12 * max(abs(epsilon_x), abs(epsilon_y), 1.0)
    isotropic = abs(epsilon_x - epsilon_y) <= tolerance
    stack_available = bool(environment)
    values = np.asarray(
        [
            epsilon_r,
            epsilon_eff,
            math.log1p(max(substrate, 0.0)),
            substrate / max(scale, 1e-12),
            math.log1p(max(metal, 0.0)),
            metal / max(scale, 1e-12),
            math.log1p(max(vacuum, 0.0)),
            vacuum / max(scale, 1e-12),
            epsilon_x,
            epsilon_y,
            epsilon_z,
            float(isotropic),
            float(stack_available),
            float(ground_available),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return values, {
        "relative_permittivity": epsilon_r,
        "effective_relative_permittivity": epsilon_eff,
        "substrate_thickness_um": substrate,
        "metal_thickness_um": metal,
        "vacuum_height_um": vacuum,
    }


def _bem_proxy(
    roles: Sequence[Any | Sequence[Any]],
    scale: float,
    effective_permittivity: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    centers = []
    lengths = []
    owners = []
    for role_index, role in enumerate(roles):
        points, weights = _segment_quadrature(role, BEM_SAMPLES_PER_ROLE)
        if len(points) == 0:
            continue
        centers.append(points)
        lengths.append(weights)
        owners.append(np.full(len(points), role_index, dtype=int))
    output = np.zeros(V3_ELECTROSTATIC_BLOCK_SIZE, dtype=np.float64)
    if len(centers) < 2:
        return output, {"available": False, "matrix_scaled_fF": np.zeros((3, 3))}

    points = np.vstack(centers)
    segment = np.concatenate(lengths)
    labels = np.concatenate(owners)
    role_count = int(labels.max()) + 1
    distances = _pair_distances(points, points) / max(scale, 1e-12)
    nonzero = distances[distances > 1e-12]
    reference_radius = 10.0 * max(float(np.max(nonzero)) if len(nonzero) else 1.0, 1.0)
    np.fill_diagonal(distances, 1.0)
    epsilon = VACUUM_PERMITTIVITY * max(effective_permittivity, 1e-9)
    green = np.log(reference_radius / np.maximum(distances, 1e-12)) / (2.0 * np.pi * epsilon)
    normalized_segment = np.maximum(segment / max(scale, 1e-12), 1e-12)
    np.fill_diagonal(
        green,
        (np.log(2.0 * reference_radius / normalized_segment) + 1.0) / (2.0 * np.pi * epsilon),
    )
    selector = np.stack([(labels == index).astype(float) for index in range(role_count)], axis=1)
    condition = float(np.linalg.cond(green))
    try:
        charges = np.linalg.solve(green, selector)
    except np.linalg.LinAlgError:
        charges = np.linalg.lstsq(green, selector, rcond=None)[0]
    raw_capacitance = selector.T @ charges
    raw_capacitance = 0.5 * (raw_capacitance + raw_capacitance.T)

    # Project noisy low-order BEM couplings onto the passive Maxwell-matrix
    # cone.  The edge magnitudes remain those of the deterministic solve while
    # symmetry, negative mutual entries, and zero row sums become exact.
    padded = np.zeros((3, 3), dtype=np.float64)
    active = min(role_count, 3)
    for first in range(active):
        for second in range(first + 1, active):
            coupling = abs(float(raw_capacitance[first, second]))
            padded[first, first] += coupling
            padded[second, second] += coupling
            padded[first, second] = padded[second, first] = -coupling
    scaled_fF = padded * scale * 1e-6 * 1e15
    dimensionless = padded / max(epsilon, 1e-30)
    upper = np.triu_indices(3)
    eigenvalues = np.linalg.eigvalsh(scaled_fF)
    output[:] = np.r_[
        _signed_log1p(scaled_fF[upper]),
        _signed_log1p(dimensionless[upper]),
        _signed_log1p(eigenvalues),
        _signed_log1p([np.trace(scaled_fF)]),
        math.log1p(abs(float(scaled_fF[0, 1]))),
        math.log1p(abs(float(scaled_fF[0, 2]))),
        math.log1p(abs(float(scaled_fF[1, 2]))),
        math.log1p(max(condition, 0.0)),
        float(role_count),
        math.log1p(len(points)),
        math.log1p(scale),
        1.0,
    ]
    return output, {
        "available": True,
        "matrix_scaled_fF": scaled_fF,
        "raw_matrix_per_depth": raw_capacitance,
        "condition_number": condition,
        "samples": len(points),
    }


def _metric_values(
    terminals: Sequence[Any],
    selected: Sequence[Any],
    conductor: Any,
    ground: Any | None,
    ground_lines: Sequence[Any],
    quadrature: Sequence[tuple[np.ndarray, np.ndarray]],
    pair_distances_um: np.ndarray,
    pair_weights: np.ndarray,
    scale: float,
) -> np.ndarray:
    terminal_areas = [float(shape.area) for shape in selected]
    terminal_perimeters = [float(shape.length) for shape in selected]
    total_area = sum(terminal_areas)
    total_perimeter = sum(terminal_perimeters)
    union = selected[0].union(selected[1])
    hull_area = max(float(union.convex_hull.area), 1e-12)
    components, holes = _component_and_hole_count(union)
    gap = float(selected[0].distance(selected[1]))
    centers = [np.asarray(shape.centroid.coords[0], dtype=np.float64) for shape in selected]
    center_distance = float(np.linalg.norm(centers[0] - centers[1]))
    weights = pair_weights
    median_distance = _weighted_quantile(pair_distances_um, weights, 0.5)
    p90_distance = _weighted_quantile(pair_distances_um, weights, 0.9)
    ground_gaps = [float(shape.distance(ground)) if ground is not None else 0.0 for shape in selected]
    boundary_stats = [_boundary_invariants(*samples, scale) for samples in quadrature]
    local_ground_length = float(sum(line.length for line in ground_lines))
    area_asymmetry = abs(terminal_areas[0] - terminal_areas[1]) / max(total_area, 1e-12)
    perimeter_asymmetry = abs(terminal_perimeters[0] - terminal_perimeters[1]) / max(total_perimeter, 1e-12)
    area_balance = min(terminal_areas) / max(max(terminal_areas), 1e-12)

    values = np.asarray(
        [
            math.log1p(scale),
            math.log1p(total_area),
            math.log1p(total_perimeter),
            total_perimeter / math.sqrt(max(total_area, 1e-12)),
            total_area / hull_area,
            float(len(terminals)),
            math.log1p(terminal_areas[0]),
            math.log1p(terminal_areas[1]),
            math.log1p(terminal_perimeters[0]),
            math.log1p(terminal_perimeters[1]),
            terminal_areas[0] / max(total_area, 1e-12),
            terminal_areas[1] / max(total_area, 1e-12),
            terminal_perimeters[0] / max(total_perimeter, 1e-12),
            terminal_perimeters[1] / max(total_perimeter, 1e-12),
            terminal_perimeters[0] / math.sqrt(max(terminal_areas[0], 1e-12)),
            terminal_perimeters[1] / math.sqrt(max(terminal_areas[1], 1e-12)),
            math.log1p(center_distance),
            center_distance / max(scale, 1e-12),
            math.log1p(gap),
            gap / max(scale, 1e-12),
            math.log1p(median_distance),
            median_distance / max(scale, 1e-12),
            math.log1p(p90_distance),
            p90_distance / max(scale, 1e-12),
            math.log1p(ground_gaps[0]),
            math.log1p(ground_gaps[1]),
            ground_gaps[0] / max(scale, 1e-12),
            ground_gaps[1] / max(scale, 1e-12),
            boundary_stats[0][0],
            boundary_stats[1][0],
            boundary_stats[0][1],
            boundary_stats[1][1],
            float(components),
            float(holes),
            math.log1p(local_ground_length),
            local_ground_length / max(scale, 1e-12),
            float(ground is not None),
            area_asymmetry,
            perimeter_asymmetry,
            area_balance,
        ],
        dtype=np.float64,
    )
    if len(values) != V3_METRIC_BLOCK_SIZE:  # pragma: no cover - schema guard
        raise RuntimeError(f"v3 metric block has {len(values)} values, expected {V3_METRIC_BLOCK_SIZE}.")
    _ = conductor
    return values


def encode_v3(
    gds_path: str | Path,
    design_options: Mapping[str, Any] | None = None,
    *,
    environment: Mapping[str, Any] | None = None,
    terminal_pair: tuple[int, int] = (0, 1),
    layer_roles: Mapping[tuple[int, int], str] | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Encode one terminal-pair capacitance query into 256 deterministic values.

    ``design_options`` is accepted for API continuity and recorded only in
    metadata.  It is intentionally absent from the physical vector: placement
    parameters, arbitrary schema names, and source-family vocabulary must not
    defeat the rigid-motion and cross-schema contract.
    """
    shapely = _require_shapely()
    geometry = read_layer_geometry(gds_path)
    grouped = _role_geometry(geometry, layer_roles or DEFAULT_LAYER_ROLES)
    conductor = shapely.union_all([shape for _, shape in grouped["conductor"]])

    origin = np.asarray(conductor.bounds[:2], dtype=np.float64)
    grouped = {
        role: [(key, shapely.transform(shape, lambda points: points - origin)) for key, shape in entries]
        for role, entries in grouped.items()
    }
    conductor = shapely.transform(conductor, lambda points: points - origin)
    terminals = _terminals(conductor, grouped["port"])
    first, second = terminal_pair
    if first == second or min(first, second) < 0 or max(first, second) >= len(terminals):
        raise ValueError(f"terminal_pair={terminal_pair} is invalid for {len(terminals)} discovered terminals.")
    selected = [terminals[first], terminals[second]]
    scale = math.sqrt(max(float(selected[0].area + selected[1].area), 1e-12))

    ground = _ground_geometry(grouped)
    ground_lines = _local_ground_lines(ground, conductor)
    terminal_quadrature = [_segment_quadrature(shape, INTERACTION_SAMPLES) for shape in selected]
    ground_quadrature = _segment_quadrature(ground_lines, INTERACTION_SAMPLES)
    role_quadrature = [*terminal_quadrature, ground_quadrature]

    interactions = []
    operators = []
    raw_pair_distance = np.zeros((0, 0), dtype=np.float64)
    raw_pair_weight = np.zeros((0, 0), dtype=np.float64)
    for pair_index, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
        spectrum, normalized_distance, pair_weight = _interaction_spectrum(
            role_quadrature[left], role_quadrature[right], scale
        )
        interactions.append(spectrum)
        operators.append(_operator_values(normalized_distance, pair_weight))
        if pair_index == 0:
            raw_pair_distance = normalized_distance * scale
            raw_pair_weight = pair_weight

    topology = np.zeros(V3_TOPOLOGY_BLOCK_SIZE, dtype=np.float64)
    terminal_union = selected[0].union(selected[1])
    for radius_index, normalized_radius in enumerate(V3_TOPOLOGY_RADII_NORMALIZED):
        buffered = terminal_union.buffer(float(normalized_radius * scale))
        components, holes = _component_and_hole_count(buffered)
        topology[radius_index] = math.log1p(components)
        topology[V3_TOPOLOGY_RADII + radius_index] = math.log1p(holes)

    environment_values, normalized_environment = _environment_values(environment, scale, bool(ground_lines))
    electrostatic, electrostatic_metadata = _bem_proxy(
        [selected[0], selected[1], ground_lines],
        scale,
        normalized_environment["effective_relative_permittivity"],
    )
    metrics = _metric_values(
        terminals,
        selected,
        conductor,
        ground,
        ground_lines,
        terminal_quadrature,
        raw_pair_distance,
        raw_pair_weight,
        scale,
    )

    vector = np.concatenate(
        [metrics, *interactions, *operators, topology, electrostatic, environment_values]
    ).astype(np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.shape != (V3_DIMENSIONS,):  # pragma: no cover - schema guard
        raise RuntimeError(f"capacitance-operator-v3 produced {vector.shape}, expected ({V3_DIMENSIONS},).")
    if not return_metadata:
        return vector
    metadata = {
        "terminal_count": len(terminals),
        "terminal_pair": [first, second],
        "characteristic_scale_um": scale,
        "ground_available": bool(ground_lines),
        "local_ground_boundary_length_um": float(sum(line.length for line in ground_lines)),
        "design_option_count": len(design_options or {}),
        "environment": normalized_environment,
        "electrostatic_proxy": electrostatic_metadata,
    }
    return vector, metadata


def capacitance_operator_v3_schema() -> dict[str, Any]:
    """Return the experimental, self-describing v3 contract."""
    offsets = np.cumsum(
        [
            0,
            V3_METRIC_BLOCK_SIZE,
            V3_INTERACTION_BLOCK_SIZE,
            V3_OPERATOR_BLOCK_SIZE,
            V3_TOPOLOGY_BLOCK_SIZE,
            V3_ELECTROSTATIC_BLOCK_SIZE,
        ]
    )
    return {
        "model": CAPACITANCE_OPERATOR_V3_MODEL,
        "embedding_schema_version": CAPACITANCE_OPERATOR_V3_SCHEMA_VERSION,
        "dimensions": V3_DIMENSIONS,
        "experimental": True,
        "fitted_on_catalogue": False,
        "simulation_results_used": False,
        "query_conditioned": True,
        "recommended_views": {
            "fixed_stack_core": {
                "dimensions": V3_FIXED_STACK_DIMENSIONS,
                "slice": [0, V3_FIXED_STACK_DIMENSIONS],
                "use_when": "the substrate and metal stack are fixed across the study",
            },
            "complete": {
                "dimensions": V3_DIMENSIONS,
                "slice": [0, V3_DIMENSIONS],
                "use_when": "stack variables change and stack-held-out validation is available",
            },
        },
        "blocks": {
            "invariant_metrics": {
                "offset": int(offsets[0]),
                "dimensions": V3_METRIC_BLOCK_SIZE,
                "values": V3_METRIC_NAMES,
            },
            "normalized_interactions": {
                "offset": int(offsets[1]),
                "dimensions": V3_INTERACTION_BLOCK_SIZE,
                "role_pairs": ["terminal_0-terminal_1", "terminal_0-ground", "terminal_1-ground"],
                "bin_edges_distance_per_scale": V3_NORMALIZED_DISTANCE_EDGES.tolist(),
            },
            "green_operator_spectra": {
                "offset": int(offsets[2]),
                "dimensions": V3_OPERATOR_BLOCK_SIZE,
                "values_per_pair": V3_OPERATOR_VALUES_PER_PAIR,
            },
            "vector_topology": {
                "offset": int(offsets[3]),
                "dimensions": V3_TOPOLOGY_BLOCK_SIZE,
                "radii_per_scale": V3_TOPOLOGY_RADII_NORMALIZED.tolist(),
            },
            "grounded_electrostatic_proxy": {
                "offset": int(offsets[4]),
                "dimensions": V3_ELECTROSTATIC_BLOCK_SIZE,
                "values": V3_ELECTROSTATIC_NAMES,
            },
            "environment": {
                "offset": int(offsets[5]),
                "dimensions": V3_ENVIRONMENT_BLOCK_SIZE,
                "values": V3_ENVIRONMENT_NAMES,
            },
        },
        "symmetry_contract": {
            "translation": "invariant for the complete vector",
            "rotation": "invariant for isotropic in-plane material when geometry and domain transform together",
            "reflection": "invariant for parity-symmetric material and boundary conditions",
            "scale": "shape channels invariant; absolute scale and stack ratios remain explicit",
            "terminal_relabeling": "equivariant; terminal-specific channels and capacitance rows/columns permute",
            "outer_domain_frame": "invariant; only moat-facing ground rings enter physical features",
        },
        "limitations": [
            "The grounded BEM block is a scale-lifted 2-D low-fidelity proxy, not a converged 3-D solve.",
            "GDS quantization and finite boundary quadrature make arbitrary-angle equality numerical rather than bitwise.",
            "Anisotropic material orientation must be supplied and transformed with the layout.",
        ],
    }


__all__ = [
    "CAPACITANCE_OPERATOR_V3_MODEL",
    "CAPACITANCE_OPERATOR_V3_SCHEMA_VERSION",
    "V3_DIMENSIONS",
    "V3_BLOCK_SLICES",
    "V3_ELECTROSTATIC_NAMES",
    "V3_ENVIRONMENT_NAMES",
    "V3_FIXED_STACK_DIMENSIONS",
    "V3_INTERACTION_BINS",
    "V3_INTERACTION_BLOCK_SIZE",
    "V3_METRIC_NAMES",
    "V3_METRIC_BLOCK_SIZE",
    "V3_NORMALIZED_DISTANCE_EDGES",
    "V3_OPERATOR_BLOCK_SIZE",
    "V3_TOPOLOGY_BLOCK_SIZE",
    "capacitance_operator_v3_schema",
    "encode_v3",
]
