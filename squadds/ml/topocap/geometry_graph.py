"""Build topology-general numerical graphs from explicit GDS net sidecars."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .net_extraction import content_sha256, extract_sidecar_geometry

GEOMETRY_GRAPH_SCHEMA_VERSION = "topocap-geometry-graph-1.1.0"
MODEL_FEATURE_TRANSFORM_VERSION = "topocap-signed-log1p-1.0.0"
ORIENTATION_BINS = 8
BOUNDARY_SAMPLE_COUNT = 96
PROXIMITY_SCALES_UM = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
MINIMUM_REFERENCE_MARGIN_UM = 10.0
REFERENCE_WINDOW_SCALE = 0.5
TOPOLOGY_CONNECTIVITY_TOLERANCE_UM = 0.002

NODE_FEATURE_NAMES = (
    "is_reference",
    "area_um2",
    "perimeter_um",
    "bbox_width_um",
    "bbox_height_um",
    "bbox_aspect_ratio",
    "bbox_occupancy",
    "centroid_abs_dx_um",
    "centroid_abs_dy_um",
    "centroid_abs_x_fraction",
    "centroid_abs_y_fraction",
    "central_mu20_um2",
    "central_mu02_um2",
    "central_abs_mu11_um2",
    "polygon_count",
    "hole_count",
    "port_count",
    "compactness",
    "solidity",
    *(f"boundary_orientation_bin_{index}" for index in range(ORIENTATION_BINS)),
    "min_distance_to_reference_um",
    "median_boundary_distance_to_reference_um",
    "min_distance_to_layout_boundary_um",
    "median_boundary_distance_to_layout_boundary_um",
)

_EDGE_BASE_FEATURE_NAMES = (
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
    "intersection_area_um2",
    "shared_boundary_length_um",
    "absolute_log_area_ratio",
    "absolute_log_perimeter_ratio",
    "reference_incidence",
)
EDGE_FEATURE_NAMES = _EDGE_BASE_FEATURE_NAMES + tuple(
    name
    for scale in PROXIMITY_SCALES_UM
    for name in (
        f"proximity_mean_ell_{scale:g}um",
        f"proximity_length_ell_{scale:g}um",
    )
)

GLOBAL_FEATURE_NAMES = (
    "net_count",
    "reference_count",
    "layout_width_um",
    "layout_height_um",
    "layout_bbox_area_um2",
    "total_net_area_um2",
    "total_net_perimeter_um",
    "auxiliary_area_um2",
    "auxiliary_perimeter_um",
    "auxiliary_polygon_count",
)

_NODE_SIGNED_LOG1P_FEATURES = frozenset(
    {
        "area_um2",
        "perimeter_um",
        "bbox_width_um",
        "bbox_height_um",
        "bbox_aspect_ratio",
        "centroid_abs_dx_um",
        "centroid_abs_dy_um",
        "central_mu20_um2",
        "central_mu02_um2",
        "central_abs_mu11_um2",
        "polygon_count",
        "hole_count",
        "port_count",
        "min_distance_to_reference_um",
        "median_boundary_distance_to_reference_um",
        "min_distance_to_layout_boundary_um",
        "median_boundary_distance_to_layout_boundary_um",
    }
)
_EDGE_SIGNED_LOG1P_FEATURES = frozenset(
    {
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
        "intersection_area_um2",
        "shared_boundary_length_um",
        *(f"proximity_length_ell_{scale:g}um" for scale in PROXIMITY_SCALES_UM),
    }
)
_GLOBAL_SIGNED_LOG1P_FEATURES = frozenset(
    {
        "layout_width_um",
        "layout_height_um",
        "layout_bbox_area_um2",
        "total_net_area_um2",
        "total_net_perimeter_um",
        "auxiliary_area_um2",
        "auxiliary_perimeter_um",
        "auxiliary_polygon_count",
    }
)


def _model_feature_names(names: Sequence[str], signed_log1p_names: frozenset[str]) -> tuple[str, ...]:
    return tuple(f"signed_log1p[{name}]" if name in signed_log1p_names else f"identity[{name}]" for name in names)


MODEL_NODE_FEATURE_NAMES = _model_feature_names(NODE_FEATURE_NAMES, _NODE_SIGNED_LOG1P_FEATURES)
MODEL_EDGE_FEATURE_NAMES = _model_feature_names(EDGE_FEATURE_NAMES, _EDGE_SIGNED_LOG1P_FEATURES)
MODEL_GLOBAL_FEATURE_NAMES = _model_feature_names(GLOBAL_FEATURE_NAMES, _GLOBAL_SIGNED_LOG1P_FEATURES)


def _require_shapely():
    try:
        import shapely
        from shapely import affinity
        from shapely.geometry.polygon import orient
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("TopoCap geometry descriptors require shapely>=2.0.") from exc
    return shapely, affinity, orient, unary_union


def _iter_polygons(geometry: Any):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for member in geometry.geoms:
            yield from _iter_polygons(member)


def _rings(geometry: Any):
    for polygon in _iter_polygons(geometry):
        yield np.asarray(polygon.exterior.coords, dtype=np.float64)
        for interior in polygon.interiors:
            yield np.asarray(interior.coords, dtype=np.float64)


def _segments(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = []
    vectors = []
    lengths = []
    for coordinates in _rings(geometry):
        if len(coordinates) < 2:
            continue
        segment_vectors = np.diff(coordinates, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        nonzero = segment_lengths > 1.0e-12
        if np.any(nonzero):
            starts.append(coordinates[:-1][nonzero])
            vectors.append(segment_vectors[nonzero])
            lengths.append(segment_lengths[nonzero])
    if not starts:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return np.vstack(starts), np.vstack(vectors), np.concatenate(lengths)


def _sample_boundary(geometry: Any, maximum_samples: int) -> tuple[np.ndarray, np.ndarray]:
    starts, vectors, lengths = _segments(geometry)
    total_length = float(lengths.sum())
    if total_length <= 0.0:
        centroid = geometry.centroid
        return np.asarray([[centroid.x, centroid.y]], dtype=np.float64), np.zeros(1, dtype=np.float64)
    sample_count = min(maximum_samples, max(16, int(math.ceil(total_length / 0.5))))
    targets = (np.arange(sample_count, dtype=np.float64) + 0.5) * total_length / sample_count
    cumulative = np.cumsum(lengths)
    segment_indices = np.searchsorted(cumulative, targets, side="right")
    previous = np.concatenate(([0.0], cumulative[:-1]))[segment_indices]
    fractions = (targets - previous) / lengths[segment_indices]
    points = starts[segment_indices] + vectors[segment_indices] * fractions[:, None]
    angles = np.mod(
        np.arctan2(vectors[segment_indices, 1], vectors[segment_indices, 0]),
        np.pi,
    )
    return points, angles


def _distances(points: np.ndarray, geometry: Any) -> np.ndarray:
    shapely, _, _, _ = _require_shapely()
    point_geometries = shapely.points(points[:, 0], points[:, 1])
    values = np.asarray(shapely.distance(point_geometries, geometry), dtype=np.float64)
    return np.maximum(values, 0.0)


def _orientation_histogram(geometry: Any) -> np.ndarray:
    _, vectors, lengths = _segments(geometry)
    if not len(lengths):
        return np.zeros(ORIENTATION_BINS, dtype=np.float64)
    angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi)
    # Assign to nearest axial bin so tiny GDS-grid perturbations around zero do
    # not send horizontal segments to opposite ends of the histogram.
    indices = np.floor(angles * ORIENTATION_BINS / np.pi + 0.5).astype(int) % ORIENTATION_BINS
    histogram = np.bincount(indices, weights=lengths, minlength=ORIENTATION_BINS).astype(np.float64)
    histogram /= max(float(histogram.sum()), np.finfo(np.float64).tiny)
    # Reflection maps theta to -theta. Averaging that orbit removes arbitrary
    # mirror conventions while retaining axial boundary structure.
    reflected = histogram[(-np.arange(ORIENTATION_BINS)) % ORIENTATION_BINS]
    invariant = 0.5 * (histogram + reflected)
    invariant /= max(float(invariant.sum()), np.finfo(np.float64).tiny)
    return invariant


def _ring_integrals(coordinates: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if len(coordinates) < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    x0 = coordinates[:-1, 0]
    y0 = coordinates[:-1, 1]
    x1 = coordinates[1:, 0]
    y1 = coordinates[1:, 1]
    cross = x0 * y1 - x1 * y0
    area = 0.5 * float(cross.sum())
    first_x = float(((x0 + x1) * cross).sum() / 6.0)
    first_y = float(((y0 + y1) * cross).sum() / 6.0)
    second_x = float(((x0 * x0 + x0 * x1 + x1 * x1) * cross).sum() / 12.0)
    second_y = float(((y0 * y0 + y0 * y1 + y1 * y1) * cross).sum() / 12.0)
    product_xy = float(((2.0 * x0 * y0 + x0 * y1 + x1 * y0 + 2.0 * x1 * y1) * cross).sum() / 24.0)
    return area, first_x, first_y, second_x, second_y, product_xy


def _central_moments(geometry: Any) -> tuple[float, float, float]:
    _, _, orient, _ = _require_shapely()
    totals = np.zeros(6, dtype=np.float64)
    for polygon in _iter_polygons(geometry):
        polygon = orient(polygon, sign=1.0)
        totals += _ring_integrals(np.asarray(polygon.exterior.coords, dtype=np.float64))
        for interior in polygon.interiors:
            totals += _ring_integrals(np.asarray(interior.coords, dtype=np.float64))
    area, first_x, first_y, second_x, second_y, product_xy = totals
    if area <= np.finfo(np.float64).tiny:
        return 0.0, 0.0, 0.0
    centroid_x = first_x / area
    centroid_y = first_y / area
    mu20 = max(second_x / area - centroid_x * centroid_x, 0.0)
    mu02 = max(second_y / area - centroid_y * centroid_y, 0.0)
    mu11 = product_xy / area - centroid_x * centroid_y
    return float(mu20), float(mu02), float(mu11)


def _canonical_rotation(nets: Sequence[Mapping[str, Any]]) -> tuple[float, tuple[float, float]]:
    _, _, _, unary_union = _require_shapely()
    non_reference = [entry["geometry"] for entry in nets if not entry["is_reference"]]
    basis = unary_union(non_reference or [entry["geometry"] for entry in nets])
    centroid = basis.centroid
    mu20, mu02, mu11 = _central_moments(basis)
    covariance = np.asarray([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(eigenvalues[-1]), 1.0)
    if float(eigenvalues[-1] - eigenvalues[0]) <= 1.0e-10 * scale:
        angle = 0.0
    else:
        axis = eigenvectors[:, -1]
        angle = math.atan2(float(axis[1]), float(axis[0]))
    return angle, (float(centroid.x), float(centroid.y))


def _rotate_extracted(extracted: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    _, affinity, _, _ = _require_shapely()
    angle, origin = _canonical_rotation(extracted["nets"])
    degrees = -math.degrees(angle)
    nets = []
    for entry in extracted["nets"]:
        nets.append(
            {
                **entry,
                "geometry": affinity.rotate(entry["geometry"], degrees, origin=origin),
                "port_geometries": [
                    affinity.rotate(geometry, degrees, origin=origin) for geometry in entry["port_geometries"]
                ],
            }
        )
    auxiliary = [
        {
            **entry,
            "geometry": affinity.rotate(entry["geometry"], degrees, origin=origin),
        }
        for entry in extracted["auxiliary_geometry"]
    ]
    return nets, auxiliary, angle


def _active_region_reference_nets(
    nets: Sequence[Mapping[str, Any]],
    *,
    minimum_reference_margin_um: float,
    reference_window_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Crop reference conductors around active, non-reference geometry.

    Port markers participate in the active envelope because they identify the
    electrically relevant terminal extent. Auxiliary etch never participates.
    """
    if not math.isfinite(minimum_reference_margin_um) or minimum_reference_margin_um < 0.0:
        raise ValueError("minimum_reference_margin_um must be finite and non-negative.")
    if not math.isfinite(reference_window_scale) or reference_window_scale < 0.0:
        raise ValueError("reference_window_scale must be finite and non-negative.")
    if minimum_reference_margin_um == 0.0 and reference_window_scale == 0.0:
        raise ValueError("At least one reference-window margin control must be positive.")

    shapely, _, _, unary_union = _require_shapely()
    active_members = [entry["geometry"] for entry in nets if not entry["is_reference"]]
    active_members.extend(port for entry in nets for port in entry["port_geometries"] if not entry["is_reference"])
    if not active_members:
        raise ValueError("Reference localization requires at least one non-reference conductor.")
    active_geometry = unary_union(active_members)
    if active_geometry.is_empty:
        raise ValueError("Non-reference conductors and port markers resolve to empty geometry.")

    active_min_x, active_min_y, active_max_x, active_max_y = active_geometry.envelope.bounds
    active_width = float(active_max_x - active_min_x)
    active_height = float(active_max_y - active_min_y)
    margin_um = max(
        float(minimum_reference_margin_um),
        float(reference_window_scale) * max(active_width, active_height),
    )
    window = shapely.box(
        active_min_x - margin_um,
        active_min_y - margin_um,
        active_max_x + margin_um,
        active_max_y + margin_um,
    )

    localized: list[dict[str, Any]] = []
    reference_audit = []
    full_references = []
    local_references = []
    for entry in nets:
        if not entry["is_reference"]:
            localized.append(dict(entry))
            continue
        full_geometry = entry["geometry"]
        local_parts = list(_iter_polygons(full_geometry.intersection(window)))
        local_geometry = unary_union(local_parts)
        if local_geometry.is_empty or float(local_geometry.area) <= np.finfo(np.float64).tiny:
            raise ValueError(
                f"Reference net {entry['net_id']!r} has no polygonal intersection with the active-region window."
            )
        full_area = float(full_geometry.area)
        local_area = float(local_geometry.area)
        localized.append({**entry, "geometry": local_geometry})
        full_references.append(full_geometry)
        local_references.append(local_geometry)
        reference_audit.append(
            {
                "net_id": str(entry["net_id"]),
                "full_area_um2": full_area,
                "local_area_um2": local_area,
                "local_to_full_area_fraction": local_area / max(full_area, np.finfo(np.float64).tiny),
            }
        )

    full_reference_area = float(unary_union(full_references).area) if full_references else 0.0
    local_reference_area = float(unary_union(local_references).area) if local_references else 0.0
    audit = {
        "method": "active_non_reference_envelope",
        "active_bounds_um": [float(value) for value in active_geometry.envelope.bounds],
        "window_bounds_um": [float(value) for value in window.bounds],
        "active_width_um": active_width,
        "active_height_um": active_height,
        "margin_um": margin_um,
        "minimum_reference_margin_um": float(minimum_reference_margin_um),
        "reference_window_scale": float(reference_window_scale),
        "full_reference_area_um2": full_reference_area,
        "local_reference_area_um2": local_reference_area,
        "local_to_full_reference_area_fraction": (
            local_reference_area / max(full_reference_area, np.finfo(np.float64).tiny) if full_references else None
        ),
        "references": reference_audit,
    }
    return localized, audit


def _topology_stable_geometry(geometry: Any) -> Any:
    # Closing at two nanometres removes topology changes caused solely by
    # rotating and requantizing polygons on the GDS grid.
    return geometry.buffer(TOPOLOGY_CONNECTIVITY_TOLERANCE_UM, join_style=2).buffer(
        -TOPOLOGY_CONNECTIVITY_TOLERANCE_UM, join_style=2
    )


def _geometry_counts(geometry: Any) -> tuple[int, int]:
    stable_geometry = _topology_stable_geometry(geometry)
    polygons = list(_iter_polygons(stable_geometry))
    return len(polygons), sum(len(polygon.interiors) for polygon in polygons)


def _node_features(
    entry: Mapping[str, Any],
    *,
    layout_bounds: tuple[float, float, float, float],
    layout_boundary: Any,
    reference_geometry: Any,
    boundary_sample_count: int,
) -> list[float]:
    geometry = entry["geometry"]
    minimum_x, minimum_y, maximum_x, maximum_y = geometry.bounds
    width = max(float(maximum_x - minimum_x), 0.0)
    height = max(float(maximum_y - minimum_y), 0.0)
    layout_min_x, layout_min_y, layout_max_x, layout_max_y = layout_bounds
    layout_width = max(float(layout_max_x - layout_min_x), np.finfo(np.float64).tiny)
    layout_height = max(float(layout_max_y - layout_min_y), np.finfo(np.float64).tiny)
    layout_center_x = 0.5 * (layout_min_x + layout_max_x)
    layout_center_y = 0.5 * (layout_min_y + layout_max_y)
    centroid = geometry.centroid
    centroid_dx = abs(float(centroid.x) - layout_center_x)
    centroid_dy = abs(float(centroid.y) - layout_center_y)
    area = float(geometry.area)
    perimeter = float(geometry.length)
    polygon_count, hole_count = _geometry_counts(geometry)
    samples, _ = _sample_boundary(geometry, boundary_sample_count)

    if entry["is_reference"]:
        reference_minimum = 0.0
        reference_median = 0.0
    else:
        reference_minimum = float(geometry.distance(reference_geometry))
        reference_median = float(np.median(_distances(samples, reference_geometry)))
    boundary_minimum = float(geometry.distance(layout_boundary))
    boundary_median = float(np.median(_distances(samples, layout_boundary)))
    mu20, mu02, mu11 = _central_moments(geometry)
    convex_area = float(geometry.convex_hull.area)
    compactness = 4.0 * np.pi * area / max(perimeter * perimeter, np.finfo(np.float64).tiny)
    orientation = _orientation_histogram(geometry)
    values = [
        float(bool(entry["is_reference"])),
        area,
        perimeter,
        width,
        height,
        max(width, height) / max(min(width, height), np.finfo(np.float64).tiny),
        area / max(width * height, np.finfo(np.float64).tiny),
        centroid_dx,
        centroid_dy,
        2.0 * centroid_dx / layout_width,
        2.0 * centroid_dy / layout_height,
        mu20,
        mu02,
        abs(mu11),
        float(polygon_count),
        float(hole_count),
        float(sum(_geometry_counts(port)[0] for port in entry["port_geometries"])),
        compactness,
        area / max(convex_area, np.finfo(np.float64).tiny),
        *orientation.tolist(),
        reference_minimum,
        reference_median,
        boundary_minimum,
        boundary_median,
    ]
    return [float(value) for value in values]


def _bbox_overlap(first: Any, second: Any) -> tuple[float, float, float, float]:
    first_min_x, first_min_y, first_max_x, first_max_y = first.bounds
    second_min_x, second_min_y, second_max_x, second_max_y = second.bounds
    overlap_x = max(0.0, min(first_max_x, second_max_x) - max(first_min_x, second_min_x))
    overlap_y = max(0.0, min(first_max_y, second_max_y) - max(first_min_y, second_min_y))
    first_width = max(first_max_x - first_min_x, np.finfo(np.float64).tiny)
    second_width = max(second_max_x - second_min_x, np.finfo(np.float64).tiny)
    first_height = max(first_max_y - first_min_y, np.finfo(np.float64).tiny)
    second_height = max(second_max_y - second_min_y, np.finfo(np.float64).tiny)
    return (
        float(overlap_x),
        float(overlap_y),
        float(overlap_x / min(first_width, second_width)),
        float(overlap_y / min(first_height, second_height)),
    )


def _edge_features(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    boundary_sample_count: int,
) -> list[float]:
    first_geometry = first["geometry"]
    second_geometry = second["geometry"]
    first_points, _ = _sample_boundary(first_geometry, boundary_sample_count)
    second_points, _ = _sample_boundary(second_geometry, boundary_sample_count)
    sampled_distances = np.concatenate(
        (
            _distances(first_points, second_geometry),
            _distances(second_points, first_geometry),
        )
    )
    quantiles = np.quantile(sampled_distances, (0.10, 0.25, 0.50, 0.75, 0.90))
    first_centroid = first_geometry.centroid
    second_centroid = second_geometry.centroid
    dx = abs(float(first_centroid.x - second_centroid.x))
    dy = abs(float(first_centroid.y - second_centroid.y))
    first_orientation = _orientation_histogram(first_geometry)
    second_orientation = _orientation_histogram(second_geometry)
    orientation_cosine = float(
        np.dot(first_orientation, second_orientation)
        / max(
            float(np.linalg.norm(first_orientation) * np.linalg.norm(second_orientation)),
            np.finfo(np.float64).tiny,
        )
    )
    orientation_overlap = float(np.minimum(first_orientation, second_orientation).sum())
    orientation_l1_similarity = 1.0 - 0.5 * float(np.abs(first_orientation - second_orientation).sum())
    first_area = max(float(first_geometry.area), np.finfo(np.float64).tiny)
    second_area = max(float(second_geometry.area), np.finfo(np.float64).tiny)
    first_perimeter = max(float(first_geometry.length), np.finfo(np.float64).tiny)
    second_perimeter = max(float(second_geometry.length), np.finfo(np.float64).tiny)
    values = [
        float(first_geometry.distance(second_geometry)),
        *quantiles.tolist(),
        dx,
        dy,
        math.hypot(dx, dy),
        *_bbox_overlap(first_geometry, second_geometry),
        orientation_cosine,
        orientation_overlap,
        orientation_l1_similarity,
        float(first_geometry.intersection(second_geometry).area),
        float(first_geometry.boundary.intersection(second_geometry.boundary).length),
        abs(math.log(first_area / second_area)),
        abs(math.log(first_perimeter / second_perimeter)),
        float(bool(first["is_reference"])) + float(bool(second["is_reference"])),
    ]
    coupled_length_scale = math.sqrt(first_perimeter * second_perimeter)
    for scale_um in PROXIMITY_SCALES_UM:
        proximity = float(np.exp(-np.minimum(sampled_distances / scale_um, 700.0)).mean())
        values.extend((proximity, proximity * coupled_length_scale))
    return [float(value) for value in values]


def _global_features(
    nets: Sequence[Mapping[str, Any]], auxiliary: Sequence[Mapping[str, Any]], layout: Any
) -> list[float]:
    minimum_x, minimum_y, maximum_x, maximum_y = layout.bounds
    auxiliary_geometries = [_topology_stable_geometry(entry["geometry"]) for entry in auxiliary]
    return [
        float(len(nets)),
        float(sum(bool(entry["is_reference"]) for entry in nets)),
        float(maximum_x - minimum_x),
        float(maximum_y - minimum_y),
        float((maximum_x - minimum_x) * (maximum_y - minimum_y)),
        float(sum(entry["geometry"].area for entry in nets)),
        float(sum(entry["geometry"].length for entry in nets)),
        float(sum(geometry.area for geometry in auxiliary_geometries)),
        float(sum(geometry.length for geometry in auxiliary_geometries)),
        float(sum(_geometry_counts(geometry)[0] for geometry in auxiliary_geometries)),
    ]


def _transform_named_features(
    values: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    feature_names: Sequence[str],
    signed_log1p_names: frozenset[str],
) -> np.ndarray:
    transformed = np.array(values, dtype=np.float64, copy=True)
    if transformed.ndim not in {1, 2} or transformed.shape[-1] != len(feature_names):
        raise ValueError(
            f"Expected a feature array ending in {len(feature_names)} columns; received {transformed.shape}."
        )
    if not np.isfinite(transformed).all():
        raise ValueError("Raw geometry features must contain only finite values.")
    indices = [index for index, name in enumerate(feature_names) if name in signed_log1p_names]
    if indices:
        selected = transformed[..., indices]
        transformed[..., indices] = np.sign(selected) * np.log1p(np.abs(selected))
    return transformed


def model_feature_arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return deterministic, scale-compressed arrays used by TopoCap models.

    The record itself intentionally keeps raw physical descriptors for audit,
    plotting, and future transforms. Bounded ratios, orientation histograms,
    topology indicators, and proximity means remain exactly unchanged.
    """
    return {
        "node_features": _transform_named_features(
            record["node_features"],
            feature_names=NODE_FEATURE_NAMES,
            signed_log1p_names=_NODE_SIGNED_LOG1P_FEATURES,
        ),
        "edge_features": _transform_named_features(
            record["edge_features"],
            feature_names=EDGE_FEATURE_NAMES,
            signed_log1p_names=_EDGE_SIGNED_LOG1P_FEATURES,
        ),
        "global_features": _transform_named_features(
            record["global_features"],
            feature_names=GLOBAL_FEATURE_NAMES,
            signed_log1p_names=_GLOBAL_SIGNED_LOG1P_FEATURES,
        ),
    }


def build_geometry_graph_record(
    path: str | Path,
    sidecar: Mapping[str, Any],
    *,
    capacitance_matrix_ff: Sequence[Sequence[float]] | np.ndarray | None = None,
    parameter_names: Sequence[str] = (),
    parameter_values: Sequence[float] = (),
    parameter_features: Sequence[Sequence[float]] = (),
    metadata: Mapping[str, Any] | None = None,
    inventory: Mapping[str, Any] | None = None,
    canonicalize_rotation: bool = True,
    boundary_sample_count: int = BOUNDARY_SAMPLE_COUNT,
    minimum_reference_margin_um: float = MINIMUM_REFERENCE_MARGIN_UM,
    reference_window_scale: float = REFERENCE_WINDOW_SCALE,
) -> dict[str, Any]:
    """Build one deterministic complete-net graph record from a GDS artifact."""
    if boundary_sample_count < 16:
        raise ValueError("boundary_sample_count must be at least 16.")
    extracted = extract_sidecar_geometry(path, sidecar, inventory=inventory)
    if canonicalize_rotation:
        nets, auxiliary, canonical_angle = _rotate_extracted(extracted)
    else:
        nets = list(extracted["nets"])
        auxiliary = list(extracted["auxiliary_geometry"])
        canonical_angle = 0.0
    nets, reference_crop = _active_region_reference_nets(
        nets,
        minimum_reference_margin_um=minimum_reference_margin_um,
        reference_window_scale=reference_window_scale,
    )
    _, _, _, unary_union = _require_shapely()
    # Auxiliary etch remains an explicit summary channel but cannot redefine
    # the physical extent used by conductor descriptors.
    layout = unary_union([entry["geometry"] for entry in nets])
    layout_boundary = layout.envelope.boundary
    references = [entry["geometry"] for entry in nets if entry["is_reference"]]
    reference_geometry = unary_union(references) if references else layout.envelope.boundary

    node_features = [
        _node_features(
            entry,
            layout_bounds=layout.bounds,
            layout_boundary=layout_boundary,
            reference_geometry=reference_geometry,
            boundary_sample_count=boundary_sample_count,
        )
        for entry in nets
    ]
    edge_pairs = [(first, second) for first in range(len(nets)) for second in range(first + 1, len(nets))]
    edge_features = [
        _edge_features(nets[first], nets[second], boundary_sample_count=boundary_sample_count)
        for first, second in edge_pairs
    ]
    matrix = None
    if capacitance_matrix_ff is not None:
        matrix_array = np.asarray(capacitance_matrix_ff, dtype=np.float64)
        if matrix_array.shape != (len(nets), len(nets)) or not np.isfinite(matrix_array).all():
            raise ValueError("capacitance_matrix_ff must be a finite square matrix matching the sidecar node count.")
        matrix = matrix_array.tolist()

    record: dict[str, Any] = {
        "schema_version": GEOMETRY_GRAPH_SCHEMA_VERSION,
        "gds_sha256": extracted["inventory"]["sha256"],
        "sidecar_sha256": extracted["sidecar"]["sidecar_sha256"],
        "net_ids": [entry["net_id"] for entry in nets],
        "node_features": node_features,
        "edge_index": [
            [pair[0] for pair in edge_pairs],
            [pair[1] for pair in edge_pairs],
        ],
        "edge_features": edge_features,
        "global_features": _global_features(nets, auxiliary, layout),
        "feature_names": {
            "raw_node": list(NODE_FEATURE_NAMES),
            "raw_edge": list(EDGE_FEATURE_NAMES),
            "raw_global": list(GLOBAL_FEATURE_NAMES),
            "model_node": list(MODEL_NODE_FEATURE_NAMES),
            "model_edge": list(MODEL_EDGE_FEATURE_NAMES),
            "model_global": list(MODEL_GLOBAL_FEATURE_NAMES),
        },
        "parameter_names": [str(name) for name in parameter_names],
        "parameter_values": [float(value) for value in parameter_values],
        "parameter_features": [[float(value) for value in row] for row in parameter_features],
        "capacitance_matrix_ff": matrix,
        "descriptor_config": {
            "coordinate_unit": "um",
            "capacitance_unit": "fF",
            "canonicalize_rotation": bool(canonicalize_rotation),
            "boundary_sample_count": int(boundary_sample_count),
            "orientation_bins": ORIENTATION_BINS,
            "proximity_scales_um": list(PROXIMITY_SCALES_UM),
            "topology_connectivity_tolerance_um": TOPOLOGY_CONNECTIVITY_TOLERANCE_UM,
            "reference_crop_method": "active_non_reference_envelope",
            "minimum_reference_margin_um": float(minimum_reference_margin_um),
            "reference_window_scale": float(reference_window_scale),
            "auxiliary_geometry_sets_active_extent": False,
            "model_feature_transform_version": MODEL_FEATURE_TRANSFORM_VERSION,
        },
        "metadata": {
            **dict(metadata or {}),
            "canonical_rotation_rad": float(canonical_angle),
            "reference_crop": reference_crop,
        },
    }
    record["record_sha256"] = content_sha256(record)
    return record


def graph_record_arrays(record: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt raw JSON descriptors to scale-compressed model arrays."""
    if record.get("schema_version") != GEOMETRY_GRAPH_SCHEMA_VERSION:
        raise ValueError(f"Unsupported geometry graph schema: {record.get('schema_version')!r}")
    matrix = record.get("capacitance_matrix_ff")
    model_arrays = model_feature_arrays(record)
    return {
        "node_features": model_arrays["node_features"],
        "edge_index": np.asarray(record["edge_index"], dtype=np.int64),
        "edge_features": model_arrays["edge_features"],
        "global_features": model_arrays["global_features"],
        "parameter_values": np.asarray(record.get("parameter_values", []), dtype=np.float64),
        "parameter_features": np.asarray(record.get("parameter_features", []), dtype=np.float64),
        "net_ids": tuple(record["net_ids"]),
        "parameter_names": tuple(record.get("parameter_names", [])),
        "capacitance_matrix": None if matrix is None else np.asarray(matrix, dtype=np.float64),
        "metadata": {
            **dict(record.get("metadata", {})),
            "gds_sha256": record["gds_sha256"],
            "sidecar_sha256": record["sidecar_sha256"],
            "record_sha256": record["record_sha256"],
            "node_feature_names": MODEL_NODE_FEATURE_NAMES,
            "edge_feature_names": MODEL_EDGE_FEATURE_NAMES,
            "global_feature_names": MODEL_GLOBAL_FEATURE_NAMES,
            "raw_node_feature_names": NODE_FEATURE_NAMES,
            "raw_edge_feature_names": EDGE_FEATURE_NAMES,
            "raw_global_feature_names": GLOBAL_FEATURE_NAMES,
            "model_feature_transform_version": MODEL_FEATURE_TRANSFORM_VERSION,
            "capacitance_unit": "fF",
        },
    }


def to_capacitance_graph(record: Mapping[str, Any]):
    """Lazily construct the graph class owned by :mod:`topocap.schema`."""
    from .schema import CapacitanceGraph

    return CapacitanceGraph(**graph_record_arrays(record))


__all__ = [
    "BOUNDARY_SAMPLE_COUNT",
    "EDGE_FEATURE_NAMES",
    "GEOMETRY_GRAPH_SCHEMA_VERSION",
    "GLOBAL_FEATURE_NAMES",
    "MINIMUM_REFERENCE_MARGIN_UM",
    "MODEL_EDGE_FEATURE_NAMES",
    "MODEL_FEATURE_TRANSFORM_VERSION",
    "MODEL_GLOBAL_FEATURE_NAMES",
    "MODEL_NODE_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "ORIENTATION_BINS",
    "PROXIMITY_SCALES_UM",
    "REFERENCE_WINDOW_SCALE",
    "TOPOLOGY_CONNECTIVITY_TOLERANCE_UM",
    "build_geometry_graph_record",
    "graph_record_arrays",
    "model_feature_arrays",
    "to_capacitance_graph",
]
