"""Universal v2 layout embeddings anchored in absolute physical units.

``universal-geometry-v2`` answers one question that v0 and v1 cannot: how does a
contributor who has never seen this catalogue produce a vector that is directly
comparable to it?

Both earlier standards are fit-on-write.  Their normalization statistics, and in
v1 the selected spectral frequencies, are derived from whichever rows happen to
be written together, so an outside group running the same builder lands in a
different space even though the vectors share a length.  v2 removes that
coupling: every coordinate is a physical measurement in micrometers, inverse
micrometers, or farads per meter, binned against frozen edges.  ``encode`` is a
pure function of one GDS file and one design-option mapping.

The second correction is to scale.  v0 and v1 crop each layout to its own
functional bounds, so a design and its exact 2x enlargement produce identical
shape blocks; only the log-area moments retain any size information.  For
capacitance the conductor separation in micrometers is the dominant quantity, so
v2 measures distances directly and bins them on an absolute logarithmic grid.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

UNIVERSAL_V2_MODEL = "universal-geometry-v2"
UNIVERSAL_V2_SCHEMA_VERSION = "2.0.0"

MAX_TERMINALS = 4
TERMINAL_PAIRS = [(i, j) for i in range(MAX_TERMINALS) for j in range(i + 1, MAX_TERMINALS)]

COUPLING_BINS = 24
GROUND_BINS = 12
SHAPE_BINS = 32
FOURIER_HARMONICS = 16
TOPOLOGY_RADII = 16

#: Frozen logarithmic distance grid in micrometers.  These edges are part of the
#: published standard: changing them changes the meaning of every coordinate.
DISTANCE_RANGE_UM = (0.1, 1000.0)
COUPLING_EDGES = np.logspace(*np.log10(DISTANCE_RANGE_UM), COUPLING_BINS + 1)
GROUND_EDGES = np.logspace(*np.log10(DISTANCE_RANGE_UM), GROUND_BINS + 1)
SHAPE_EDGES = np.logspace(*np.log10(DISTANCE_RANGE_UM), SHAPE_BINS + 1)
TOPOLOGY_RADII_UM = np.logspace(np.log10(0.1), np.log10(200.0), TOPOLOGY_RADII)

METRIC_BLOCK_SIZE = 48
COUPLING_BLOCK_SIZE = len(TERMINAL_PAIRS) * COUPLING_BINS + MAX_TERMINALS * GROUND_BINS
SHAPE_BLOCK_SIZE = 4 * SHAPE_BINS
PARAMETER_BLOCK_SIZE = 96
PHYSICS_BLOCK_SIZE = 16 + 2 * TOPOLOGY_RADII
V2_DIMENSIONS = METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE + PARAMETER_BLOCK_SIZE + PHYSICS_BLOCK_SIZE

RASTER_SIZE = 256
VACUUM_PERMITTIVITY = 8.8541878128e-12

DEFAULT_LAYER_ROLES: dict[tuple[int, int], str] = {
    (1, 10): "conductor",
    (1, 11): "etch",
    (1, 0): "domain",
    (2, 0): "port",
    (3, 0): "port",
    (4, 0): "port",
    (5, 0): "port",
}

METRIC_NAMES = [
    "log1p_bbox_width_um",
    "log1p_bbox_height_um",
    "log_bbox_aspect_ratio",
    "log1p_bbox_diagonal_um",
    "log1p_conductor_area_um2",
    "log1p_conductor_perimeter_um",
    "log1p_etch_area_um2",
    "log1p_etch_perimeter_um",
    "log1p_port_area_um2",
    "log1p_domain_area_um2",
    "terminal_count",
    "log1p_polygon_count",
    "log1p_vertex_count",
    "log1p_layer_count",
    "isoperimetric_compactness",
    "conductor_fill_fraction",
    "log1p_conductor_width_p05_um",
    "log1p_conductor_width_p25_um",
    "log1p_conductor_width_p50_um",
    "log1p_conductor_width_p75_um",
    "log1p_conductor_width_p95_um",
    "log1p_conductor_width_max_um",
    "log1p_terminal_0_area_um2",
    "log1p_terminal_1_area_um2",
    "log1p_terminal_2_area_um2",
    "log1p_terminal_3_area_um2",
    "log1p_terminal_0_perimeter_um",
    "log1p_terminal_1_perimeter_um",
    "log1p_terminal_2_perimeter_um",
    "log1p_terminal_3_perimeter_um",
    "log1p_minimum_pair_gap_um",
    "log1p_median_pair_gap_um",
    "log1p_maximum_pair_gap_um",
    "log1p_primary_inverse_gap_integral",
    "log1p_total_inverse_gap_integral",
    "log1p_total_logarithmic_gap_integral",
    "horizontal_symmetry",
    "vertical_symmetry",
    "diagonal_symmetry",
    "rotational_symmetry",
    "conductor_eccentricity",
    "conductor_mu20",
    "conductor_mu02",
    "conductor_mu11",
    "log1p_area_per_terminal_um2",
    "log1p_perimeter_per_terminal_um",
    "log1p_functional_area_um2",
    "largest_terminal_area_fraction",
]

PARAMETER_DIMENSION_CLASSES = ("length", "count", "angle", "boolean", "other")
PARAMETER_STATISTIC_SIZES = {"length": 24, "count": 12, "angle": 8, "boolean": 4, "other": 12}
PARAMETER_HASH_DIMENSIONS = PARAMETER_BLOCK_SIZE - sum(PARAMETER_STATISTIC_SIZES.values())

_NUMBER_WITH_UNIT = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]*)\s*$")
_LENGTH_UNITS = {"um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}
_ANGLE_UNITS = {"deg": 1.0, "degree": 1.0, "degrees": 1.0, "rad": 180.0 / math.pi}
_LENGTH_TOKENS = (
    "length",
    "width",
    "gap",
    "radius",
    "offset",
    "pos",
    "size",
    "thickness",
    "pitch",
    "spacing",
    "height",
    "extent",
    "margin",
    "buffer",
)
_COUNT_TOKENS = ("count", "number", "num_", "_num", "n_", "fingers", "turns", "segments", "layer")
_ANGLE_TOKENS = ("angle", "orientation", "rotation", "theta", "phi")


def _require(module: str, extra: str):
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(f"universal-geometry-v2 requires {module}: {extra}") from exc


# ---------------------------------------------------------------------------
# Geometry ingestion
# ---------------------------------------------------------------------------


def read_layer_geometry(path: str | Path) -> dict[tuple[int, int], Any]:
    """Return one Shapely geometry per ``(layer, datatype)`` pair, holes included.

    ``parse_gds_polygons`` in :mod:`squadds.layouts.manifest` returns only outer
    hulls, which is adequate for bitmaps but silently fills etched interiors.
    v2 measures conductor separation, so hole fidelity matters.
    """
    kdb = _require("klayout.db", "uv sync --extra gds")
    shapely = _require("shapely", "uv sync --extra gds")

    layout = kdb.Layout()
    layout.read(str(path))
    top_cell = layout.top_cell()
    if top_cell is None:
        raise ValueError(f"GDS file has no top cell: {path}")
    dbu = float(layout.dbu)

    geometry: dict[tuple[int, int], Any] = {}
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        region = kdb.Region(top_cell.begin_shapes_rec(layer_index))
        region.merge()
        polygons = []
        for polygon in region.each():
            shell = [(point.x * dbu, point.y * dbu) for point in polygon.each_point_hull()]
            if len(shell) < 3:
                continue
            holes = []
            for hole_index in range(polygon.holes()):
                ring = [(point.x * dbu, point.y * dbu) for point in polygon.each_point_hole(hole_index)]
                if len(ring) >= 3:
                    holes.append(ring)
            polygons.append(shapely.Polygon(shell, holes))
        if polygons:
            geometry[(int(info.layer), int(info.datatype))] = shapely.union_all(polygons)
    return geometry


def _role_geometry(
    geometry: Mapping[tuple[int, int], Any],
    layer_roles: Mapping[tuple[int, int], str] | None,
) -> dict[str, list[Any]]:
    shapely = _require("shapely", "uv sync --extra gds")
    roles = dict(DEFAULT_LAYER_ROLES if layer_roles is None else layer_roles)
    grouped: dict[str, list[Any]] = {"conductor": [], "etch": [], "port": [], "domain": []}
    for key, shape in geometry.items():
        role = roles.get(key)
        if role is None:
            # An unmapped layer is treated as conductor unless it is the usual
            # simulation-domain layer.  This keeps foreign layouts encodable.
            role = None if key == (1, 0) else "conductor"
        if role in grouped:
            grouped[role].append((key, shape))
    if not grouped["conductor"]:
        raise ValueError("No conductor geometry found; supply layer_roles for this layout.")
    _ = shapely
    return grouped


def _terminals(conductor: Any, ports: list[tuple[tuple[int, int], Any]]) -> list[Any]:
    """Split the conductor set into terminals and order them canonically.

    Port markers give a stable, tool-independent ordering when present.  Without
    them the ordering falls back to descending area, which is deterministic for
    any layout.
    """
    shapely = _require("shapely", "uv sync --extra gds")
    merged = shapely.union_all([conductor]) if conductor.geom_type == "Polygon" else conductor
    parts = list(getattr(merged, "geoms", [merged]))
    parts = [part for part in parts if part.area > 0]
    if not parts:
        raise ValueError("Conductor geometry is empty.")

    port_rank: dict[int, float] = {}
    for (layer, datatype), port_shape in sorted(ports):
        distances = [part.distance(port_shape) for part in parts]
        nearest = int(np.argmin(distances))
        port_rank.setdefault(nearest, float(layer) + float(datatype) / 1000.0)
    order = sorted(
        range(len(parts)),
        key=lambda index: (port_rank.get(index, math.inf), -parts[index].area, index),
    )
    return [parts[index] for index in order]


# ---------------------------------------------------------------------------
# Rasterization (adaptive resolution, absolute bins)
# ---------------------------------------------------------------------------


def _raster_frame(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    width = max(right - left, 1e-9)
    height = max(top - bottom, 1e-9)
    span = max(width, height) * 1.1
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (bottom + top)
    pixel = span / RASTER_SIZE
    return center_x - span / 2, center_y - span / 2, pixel, span


def _rasterize(shape: Any, frame: tuple[float, float, float, float]) -> np.ndarray:
    """Draw one geometry, honoring holes, into the shared absolute raster frame."""
    image_module = _require("PIL.Image", "uv sync --extra gds")
    draw_module = _require("PIL.ImageDraw", "uv sync --extra gds")
    origin_x, origin_y, pixel, _ = frame
    image = image_module.new("L", (RASTER_SIZE, RASTER_SIZE), 0)
    painter = draw_module.Draw(image)
    if shape is None or shape.is_empty:
        return np.zeros((RASTER_SIZE, RASTER_SIZE), dtype=bool)
    for polygon in getattr(shape, "geoms", [shape]):
        if polygon.geom_type != "Polygon":
            continue
        exterior = [((x - origin_x) / pixel, RASTER_SIZE - (y - origin_y) / pixel) for x, y in polygon.exterior.coords]
        painter.polygon(exterior, fill=255)
        for interior in polygon.interiors:
            ring = [((x - origin_x) / pixel, RASTER_SIZE - (y - origin_y) / pixel) for x, y in interior.coords]
            painter.polygon(ring, fill=0)
    return np.asarray(image, dtype=np.uint8) > 127


def _radial_profile(correlation: np.ndarray, pixel: float, edges: np.ndarray) -> np.ndarray:
    """Bin a centered autocorrelation map onto the frozen micrometer grid."""
    size = correlation.shape[0]
    coordinates = np.arange(size) - size // 2
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2) * pixel
    index = np.digitize(radius.reshape(-1), edges) - 1
    values = correlation.reshape(-1)
    profile = np.zeros(len(edges) - 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.float64)
    valid = (index >= 0) & (index < len(profile))
    np.add.at(profile, index[valid], values[valid])
    np.add.at(counts, index[valid], 1.0)
    return profile / np.maximum(counts, 1.0)


def _autocorrelation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft2(first.astype(np.float64))
    other = spectrum if second is first else np.fft.rfft2(second.astype(np.float64))
    correlation = np.fft.irfft2(spectrum * np.conj(other), s=first.shape)
    return np.fft.fftshift(correlation) / first.size


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _boundary_samples(shape: Any, target: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Sample a boundary at uniform arclength and return points with their weights."""
    shapely = _require("shapely", "uv sync --extra gds")
    boundary = shape.boundary
    length = float(boundary.length)
    if length <= 0:
        return np.zeros((0, 2)), np.zeros(0)
    count = int(min(target, max(64, target)))
    positions = (np.arange(count) + 0.5) * length / count
    points = shapely.line_interpolate_point(boundary, positions)
    coordinates = shapely.get_coordinates(points)
    weights = np.full(count, length / count)
    return coordinates, weights


def soft_histogram(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Accumulate weights onto a log-spaced grid with linear interpolation.

    Hard binning makes the spectrum discontinuous at every edge: a sample that
    drifts across an edge moves its whole weight between coordinates.  That
    turns a smooth geometric change into a jump in the embedding and makes the
    result sensitive to how a boundary happens to be sampled.  Splitting each
    sample between its two neighbouring bin centres keeps the spectrum Lipschitz
    in the measured distance.  Values outside the grid clamp to the end bins so
    no mass is discarded.
    """
    bins = len(edges) - 1
    histogram = np.zeros(bins, dtype=np.float64)
    if len(values) == 0:
        return histogram
    log_centers = 0.5 * (np.log(edges[:-1]) + np.log(edges[1:]))
    clipped = np.clip(values, edges[0], edges[-1])
    position = np.interp(np.log(np.maximum(clipped, 1e-12)), log_centers, np.arange(bins))
    lower = np.clip(np.floor(position).astype(int), 0, bins - 1)
    upper = np.clip(lower + 1, 0, bins - 1)
    fraction = position - lower
    np.add.at(histogram, lower, weights * (1.0 - fraction))
    np.add.at(histogram, upper, weights * fraction)
    return histogram


def _distance_histogram(
    coordinates: np.ndarray,
    weights: np.ndarray,
    other: Any,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary length per absolute distance bin, plus the raw distances."""
    shapely = _require("shapely", "uv sync --extra gds")
    if len(coordinates) == 0 or other is None or other.is_empty:
        return np.zeros(len(edges) - 1), np.zeros(0)
    points = shapely.points(coordinates)
    distances = shapely.distance(points, other)
    return soft_histogram(distances, weights, edges), distances


def _coupling_block(
    terminals: list[Any],
    ground: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Boundary length of each terminal at each absolute distance from another.

    For coplanar geometry the mutual capacitance is approximately an integral of
    a kernel over facing boundary length at a given separation, so a model that
    is linear in this histogram can represent the Green's function directly.
    """
    values = np.zeros(COUPLING_BLOCK_SIZE, dtype=np.float64)
    availability = np.zeros(len(TERMINAL_PAIRS) + MAX_TERMINALS, dtype=bool)
    samples = [_boundary_samples(terminal) for terminal in terminals[:MAX_TERMINALS]]

    pair_gaps: list[float] = []
    inverse_integrals: list[float] = []
    logarithmic_integrals: list[float] = []
    primary_inverse = 0.0
    for slot, (first, second) in enumerate(TERMINAL_PAIRS):
        offset = slot * COUPLING_BINS
        if first >= len(terminals) or second >= len(terminals):
            continue
        forward, forward_distances = _distance_histogram(*samples[first], terminals[second], COUPLING_EDGES)
        backward, backward_distances = _distance_histogram(*samples[second], terminals[first], COUPLING_EDGES)
        values[offset : offset + COUPLING_BINS] = np.log1p(0.5 * (forward + backward))
        availability[slot] = True

        distances = np.concatenate([forward_distances, backward_distances])
        weights = np.concatenate([samples[first][1], samples[second][1]])
        safe = np.maximum(distances, 1e-3)
        inverse = float(np.sum(weights / safe))
        logarithmic = float(np.sum(weights * np.log(2000.0 / safe)))
        pair_gaps.append(float(np.min(safe)))
        inverse_integrals.append(inverse)
        logarithmic_integrals.append(logarithmic)
        if (first, second) == (0, 1):
            primary_inverse = inverse

    ground_offset = len(TERMINAL_PAIRS) * COUPLING_BINS
    for index in range(MAX_TERMINALS):
        offset = ground_offset + index * GROUND_BINS
        if index >= len(terminals) or ground is None or ground.is_empty:
            continue
        histogram, _ = _distance_histogram(*samples[index], ground, GROUND_EDGES)
        values[offset : offset + GROUND_BINS] = np.log1p(histogram)
        availability[len(TERMINAL_PAIRS) + index] = True

    summary = {
        "pair_gaps_um": pair_gaps,
        "inverse_gap_integrals": inverse_integrals,
        "logarithmic_gap_integrals": logarithmic_integrals,
        "primary_inverse_gap_integral": primary_inverse,
        "availability": availability,
    }
    return values, summary


def _shape_block(
    conductor_mask: np.ndarray,
    terminal_masks: list[np.ndarray],
    pixel: float,
    terminals: list[Any],
) -> np.ndarray:
    """Correlation lengths, conductor width distribution, and contour harmonics.

    All three are measured in absolute micrometers, so a design and its scaled
    copy occupy different coordinates rather than the identical ones v0 and v1
    assign them.
    """
    ndimage = _require("scipy.ndimage", "uv sync --extra gds")
    block = np.zeros(SHAPE_BLOCK_SIZE, dtype=np.float64)

    occupancy = float(conductor_mask.mean())
    if occupancy > 0:
        auto = _autocorrelation(conductor_mask, conductor_mask)
        block[0:SHAPE_BINS] = _radial_profile(auto, pixel, SHAPE_EDGES) / max(occupancy, 1e-12)
    if len(terminal_masks) >= 2 and terminal_masks[0].any() and terminal_masks[1].any():
        cross = _autocorrelation(terminal_masks[0], terminal_masks[1])
        scale = math.sqrt(max(terminal_masks[0].mean(), 1e-12) * max(terminal_masks[1].mean(), 1e-12))
        block[SHAPE_BINS : 2 * SHAPE_BINS] = _radial_profile(cross, pixel, SHAPE_EDGES) / scale

    if conductor_mask.any():
        interior = ndimage.distance_transform_edt(conductor_mask) * pixel
        widths = 2.0 * interior[conductor_mask]
        area_weights = np.full(len(widths), pixel * pixel)
        block[2 * SHAPE_BINS : 3 * SHAPE_BINS] = np.log1p(soft_histogram(widths, area_weights, SHAPE_EDGES))

    harmonics = np.zeros(FOURIER_HARMONICS, dtype=np.float64)
    scale_harmonics = np.zeros(FOURIER_HARMONICS, dtype=np.float64)
    counted = 0
    for terminal in terminals[:2]:
        coordinates, _ = _boundary_samples(terminal, target=256)
        if len(coordinates) < 8:
            continue
        signal = coordinates[:, 0] + 1j * coordinates[:, 1]
        spectrum = np.fft.fft(signal - signal.mean())
        magnitude = np.abs(spectrum[1 : FOURIER_HARMONICS + 1]) / len(signal)
        harmonics += magnitude / max(float(magnitude[0]), 1e-12)
        scale_harmonics += np.log1p(magnitude)
        counted += 1
    if counted:
        block[3 * SHAPE_BINS : 3 * SHAPE_BINS + FOURIER_HARMONICS] = harmonics / counted
        block[3 * SHAPE_BINS + FOURIER_HARMONICS : 4 * SHAPE_BINS] = scale_harmonics / counted
    return block


def _physics_block(terminals: list[Any], conductor_mask: np.ndarray, pixel: float) -> np.ndarray:
    """A two-dimensional boundary-element capacitance proxy plus dilation topology.

    The proxy ignores the substrate and every three-dimensional effect, so it is
    not a simulation.  It is a feature: it supplies the part of the map that is
    identical for every component class, leaving the learned head to fit only a
    smoother, more transferable correction.
    """
    ndimage = _require("scipy.ndimage", "uv sync --extra gds")
    block = np.zeros(PHYSICS_BLOCK_SIZE, dtype=np.float64)
    active = terminals[:MAX_TERMINALS]
    if len(active) >= 2:
        centers, lengths, owner = [], [], []
        for index, terminal in enumerate(active):
            coordinates, weights = _boundary_samples(terminal, target=160)
            centers.append(coordinates)
            lengths.append(weights)
            owner.append(np.full(len(coordinates), index))
        points = np.vstack(centers)
        segment = np.concatenate(lengths)
        labels = np.concatenate(owner)
        delta = points[:, None, :] - points[None, :, :]
        distance = np.sqrt(np.sum(delta**2, axis=2)) * 1e-6
        np.fill_diagonal(distance, 1.0)
        green = -np.log(distance) / (2.0 * math.pi * VACUUM_PERMITTIVITY)
        np.fill_diagonal(green, -(np.log(segment * 1e-6 / 2.0) - 1.0) / (2.0 * math.pi * VACUUM_PERMITTIVITY))
        selector = np.stack([(labels == index).astype(float) for index in range(len(active))], axis=1)
        try:
            charges = np.linalg.solve(green, selector)
        except np.linalg.LinAlgError:  # pragma: no cover - singular systems are rare
            charges = np.linalg.lstsq(green, selector, rcond=None)[0]
        capacitance = selector.T @ charges
        entries = []
        for i in range(MAX_TERMINALS):
            for j in range(i, MAX_TERMINALS):
                if i < len(active) and j < len(active):
                    entries.append(math.copysign(math.log1p(abs(capacitance[i, j]) * 1e15), capacitance[i, j]))
                else:
                    entries.append(0.0)
        block[:10] = entries[:10]
        eigenvalues = np.linalg.eigvalsh(0.5 * (capacitance + capacitance.T))
        for index in range(min(len(eigenvalues), 4)):
            block[10 + index] = math.copysign(math.log1p(abs(eigenvalues[index]) * 1e15), eigenvalues[index])
        block[14] = math.log1p(abs(float(np.trace(capacitance))) * 1e15)
        block[15] = float(len(active))

    # One exterior distance transform thresholds to every dilation radius at
    # once.  Iterating binary_dilation instead costs O(radius) passes per radius
    # and dominated the whole encoder.
    offset = 16
    exterior = ndimage.distance_transform_edt(~conductor_mask) * pixel
    for index, radius in enumerate(TOPOLOGY_RADII_UM):
        dilated = exterior <= radius
        _, components = ndimage.label(dilated)
        filled = ndimage.binary_fill_holes(dilated)
        _, holes = ndimage.label(filled & ~dilated)
        block[offset + index] = math.log1p(components)
        block[offset + TOPOLOGY_RADII + index] = math.log1p(holes)
    return block


# ---------------------------------------------------------------------------
# Parameter block
# ---------------------------------------------------------------------------


def classify_parameter(name: str, value: Any) -> tuple[str, float] | None:
    """Map one design option to a physical dimension class and canonical value.

    Aligning contributors by what a parameter *is* rather than what it is called
    is what makes a 28-parameter foreign schema commensurable with a 41-parameter
    local one without a name-matching table.
    """
    lowered = name.lower()
    if isinstance(value, bool):
        return "boolean", float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        unit = ""
    elif isinstance(value, str):
        match = _NUMBER_WITH_UNIT.match(value)
        if not match:
            return None
        numeric = float(match.group(1))
        unit = match.group(2).lower()
    else:
        return None

    if unit in _LENGTH_UNITS:
        return "length", numeric * _LENGTH_UNITS[unit]
    if unit in _ANGLE_UNITS:
        return "angle", numeric * _ANGLE_UNITS[unit]
    if unit:
        return "other", numeric
    if any(token in lowered for token in _ANGLE_TOKENS):
        return "angle", numeric
    if any(token in lowered for token in _COUNT_TOKENS):
        return "count", numeric
    if any(token in lowered for token in _LENGTH_TOKENS):
        return "length", numeric
    return "other", numeric


def _flatten_options(options: Mapping[str, Any], prefix: str = ""):
    for key in sorted(options):
        value = options[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_options(value, path)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                if isinstance(nested, Mapping):
                    yield from _flatten_options(nested, f"{path}[{index}]")
                else:
                    yield f"{path}[{index}]", nested
        else:
            yield path, value


def _signed_statistics(values: np.ndarray, order_count: int) -> list[float]:
    """Summaries plus order statistics, all in an absolute logarithmic scale."""
    order_count = max(0, order_count)
    if len(values) == 0:
        return [0.0] * (8 + 2 * order_count)
    signed = np.sign(values) * np.log1p(np.abs(values))
    ordered = np.sort(signed)
    smallest = list(ordered[:order_count]) + [0.0] * max(0, order_count - len(ordered))
    largest = list(ordered[::-1][:order_count]) + [0.0] * max(0, order_count - len(ordered))
    return [
        math.log1p(len(values)),
        float(np.sum(signed)),
        float(np.sum(signed) / len(values)),
        float(np.min(signed)),
        float(np.max(signed)),
        float(np.median(signed)),
        float(np.std(signed)),
        float(np.max(signed) - np.min(signed)),
        *smallest[:order_count],
        *largest[:order_count],
    ]


def parameter_block(design_options: Mapping[str, Any] | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode any parameter schema into a fixed width without a name registry."""
    block = np.zeros(PARAMETER_BLOCK_SIZE, dtype=np.float64)
    grouped: dict[str, list[float]] = {name: [] for name in PARAMETER_DIMENSION_CLASSES}
    names: list[str] = []
    classes: list[str] = []
    canonical: list[float] = []
    if design_options:
        for name, value in _flatten_options(design_options):
            classified = classify_parameter(name, value)
            if classified is None:
                continue
            dimension, numeric = classified
            grouped[dimension].append(numeric)
            names.append(name)
            classes.append(dimension)
            canonical.append(numeric)

    offset = 0
    for dimension in PARAMETER_DIMENSION_CLASSES:
        width = PARAMETER_STATISTIC_SIZES[dimension]
        order_count = max(0, (width - 8) // 2)
        statistics = _signed_statistics(np.asarray(grouped[dimension], dtype=np.float64), order_count)
        padded = (statistics + [0.0] * width)[:width]
        block[offset : offset + width] = padded
        offset += width

    for name, dimension, numeric in zip(names, classes, canonical):
        digest = hashlib.sha256(f"{dimension}:{name}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % PARAMETER_HASH_DIMENSIONS
        sign = 1.0 if digest[8] & 1 else -1.0
        block[offset + index] += sign * math.copysign(math.log1p(abs(numeric)), numeric)

    metadata = {
        "parameter_names": names,
        "parameter_classes": classes,
        "parameter_values": canonical,
        "parameter_count": len(names),
    }
    return block, metadata


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _metric_block(
    terminals: list[Any],
    grouped: dict[str, list[Any]],
    conductor: Any,
    conductor_mask: np.ndarray,
    pixel: float,
    coupling_summary: dict[str, Any],
    geometry: Mapping[tuple[int, int], Any],
) -> np.ndarray:
    ndimage = _require("scipy.ndimage", "uv sync --extra gds")
    values = np.zeros(METRIC_BLOCK_SIZE, dtype=np.float64)
    left, bottom, right, top = conductor.bounds
    width = max(right - left, 1e-9)
    height = max(top - bottom, 1e-9)

    def role_area(role: str) -> float:
        return float(sum(shape.area for _, shape in grouped[role]))

    def role_perimeter(role: str) -> float:
        return float(sum(shape.length for _, shape in grouped[role]))

    conductor_area = float(conductor.area)
    conductor_perimeter = float(conductor.length)
    vertex_count = sum(
        len(polygon.exterior.coords) + sum(len(ring.coords) for ring in polygon.interiors)
        for polygon in getattr(conductor, "geoms", [conductor])
        if polygon.geom_type == "Polygon"
    )
    polygon_count = len(list(getattr(conductor, "geoms", [conductor])))

    interior = ndimage.distance_transform_edt(conductor_mask) * pixel if conductor_mask.any() else np.zeros(1)
    widths = 2.0 * interior[conductor_mask] if conductor_mask.any() else np.zeros(1)
    percentiles = np.percentile(widths, [5, 25, 50, 75, 95]) if len(widths) else np.zeros(5)

    gaps = coupling_summary["pair_gaps_um"] or [0.0]
    inverse = coupling_summary["inverse_gap_integrals"] or [0.0]
    logarithmic = coupling_summary["logarithmic_gap_integrals"] or [0.0]

    flipped_horizontal = float(np.mean(conductor_mask == np.fliplr(conductor_mask)))
    flipped_vertical = float(np.mean(conductor_mask == np.flipud(conductor_mask)))
    flipped_diagonal = float(np.mean(conductor_mask == conductor_mask.T))
    rotated = float(np.mean(conductor_mask == np.rot90(conductor_mask, 2)))

    weights = conductor_mask.astype(float)
    total = max(float(weights.sum()), 1e-12)
    yy, xx = np.indices(conductor_mask.shape, dtype=float)
    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)
    scale = max(conductor_mask.shape[0] - 1, 1)
    mu20 = float((((xx - cx) / scale) ** 2 * weights).sum() / total)
    mu02 = float((((yy - cy) / scale) ** 2 * weights).sum() / total)
    mu11 = float((((xx - cx) / scale) * ((yy - cy) / scale) * weights).sum() / total)
    eigenvalues = np.linalg.eigvalsh(np.asarray([[mu20, mu11], [mu11, mu02]]))
    eccentricity = math.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[1], 1e-12)))

    terminal_areas = [float(terminal.area) for terminal in terminals[:MAX_TERMINALS]]
    terminal_perimeters = [float(terminal.length) for terminal in terminals[:MAX_TERMINALS]]

    entries = [
        math.log1p(width),
        math.log1p(height),
        math.log(width / height),
        math.log1p(math.hypot(width, height)),
        math.log1p(conductor_area),
        math.log1p(conductor_perimeter),
        math.log1p(role_area("etch")),
        math.log1p(role_perimeter("etch")),
        math.log1p(role_area("port")),
        math.log1p(role_area("domain")),
        float(len(terminals)),
        math.log1p(polygon_count),
        math.log1p(vertex_count),
        math.log1p(len(geometry)),
        conductor_perimeter / math.sqrt(max(conductor_area, 1e-12)),
        conductor_area / max(width * height, 1e-12),
        *[math.log1p(value) for value in percentiles],
        math.log1p(float(np.max(widths)) if len(widths) else 0.0),
        *[math.log1p(value) for value in (terminal_areas + [0.0] * MAX_TERMINALS)[:MAX_TERMINALS]],
        *[math.log1p(value) for value in (terminal_perimeters + [0.0] * MAX_TERMINALS)[:MAX_TERMINALS]],
        math.log1p(min(gaps)),
        math.log1p(float(np.median(gaps))),
        math.log1p(max(gaps)),
        math.log1p(coupling_summary["primary_inverse_gap_integral"]),
        math.log1p(sum(inverse)),
        math.log1p(max(sum(logarithmic), 0.0)),
        flipped_horizontal,
        flipped_vertical,
        flipped_diagonal,
        rotated,
        eccentricity,
        mu20,
        mu02,
        mu11,
        math.log1p(conductor_area / max(len(terminals), 1)),
        math.log1p(conductor_perimeter / max(len(terminals), 1)),
        math.log1p(conductor_area + role_area("etch")),
        (max(terminal_areas) / conductor_area) if terminal_areas and conductor_area > 0 else 0.0,
    ]
    values[: len(entries)] = entries[:METRIC_BLOCK_SIZE]
    return values


def encode(
    gds_path: str | Path,
    design_options: Mapping[str, Any] | None = None,
    *,
    layer_roles: Mapping[tuple[int, int], str] | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Encode one layout into the frozen ``universal-geometry-v2`` vector.

    This is a pure function.  No catalogue, no fitted statistics, and no other
    rows are consulted, so a contributor with a thousand designs of an unseen
    topology lands in exactly the space this repository already occupies.
    """
    shapely = _require("shapely", "uv sync --extra gds")
    geometry = read_layer_geometry(gds_path)
    grouped = _role_geometry(geometry, layer_roles)
    conductor = shapely.union_all([shape for _, shape in grouped["conductor"]])

    # Re-origin every geometry on the conductor bounds before measuring.  Each
    # feature below is translation invariant in exact arithmetic, but rasterizing
    # a layout centered at 10^4 um loses sub-pixel precision to cancellation in
    # ``x - origin``.  Normalizing first makes the invariance hold bit-for-bit.
    origin_x, origin_y, _, _ = conductor.bounds
    offset = shapely.transform(conductor, lambda points: points - np.asarray([origin_x, origin_y]))
    grouped = {
        role: [
            (key, shapely.transform(shape, lambda points: points - np.asarray([origin_x, origin_y])))
            for key, shape in entries
        ]
        for role, entries in grouped.items()
    }
    conductor = offset
    terminals = _terminals(conductor, grouped["port"])

    frame = _raster_frame(conductor.bounds)
    _, _, pixel, _ = frame
    conductor_mask = _rasterize(conductor, frame)
    terminal_masks = [_rasterize(terminal, frame) for terminal in terminals[:MAX_TERMINALS]]

    ground_parts = [shape for _, shape in grouped["etch"]] + [shape for _, shape in grouped["domain"]]
    ground = shapely.union_all(ground_parts) if ground_parts else None
    if ground is not None and ground.is_empty:
        ground = None

    coupling, coupling_summary = _coupling_block(terminals, ground)
    metrics = _metric_block(terminals, grouped, conductor, conductor_mask, pixel, coupling_summary, geometry)
    shape_features = _shape_block(conductor_mask, terminal_masks, pixel, terminals)
    physics = _physics_block(terminals, conductor_mask, pixel)
    parameters, parameter_metadata = parameter_block(design_options)

    vector = np.concatenate([metrics, coupling, shape_features, parameters, physics]).astype(np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.shape[0] != V2_DIMENSIONS:  # pragma: no cover - guarded by tests
        raise RuntimeError(f"universal-geometry-v2 produced {vector.shape[0]} dimensions, expected {V2_DIMENSIONS}.")
    if not return_metadata:
        return vector
    metadata = {
        "terminal_count": len(terminals),
        "raster_pixel_um": pixel,
        "functional_bounds_um": dict(zip(("left", "bottom", "right", "top"), conductor.bounds)),
        "coupling_availability": coupling_summary["availability"].tolist(),
        "minimum_pair_gap_um": min(coupling_summary["pair_gaps_um"]) if coupling_summary["pair_gaps_um"] else None,
        **parameter_metadata,
    }
    return vector, metadata


def universal_v2_schema() -> dict[str, Any]:
    """Return the frozen, self-describing v2 contract."""
    return {
        "model": UNIVERSAL_V2_MODEL,
        "embedding_schema_version": UNIVERSAL_V2_SCHEMA_VERSION,
        "dimensions": V2_DIMENSIONS,
        "fitted_on_catalogue": False,
        "input_contract": {
            "required": ["GDSII geometry with the published layer semantics"],
            "optional": ["native design-parameter mapping of any size"],
            "layer_roles": ["conductor", "etch", "port", "domain"],
            "simulation_results_used": False,
        },
        "blocks": {
            "physical_metrics": {"offset": 0, "dimensions": METRIC_BLOCK_SIZE, "values": METRIC_NAMES},
            "coupling_spectrum": {
                "offset": METRIC_BLOCK_SIZE,
                "dimensions": COUPLING_BLOCK_SIZE,
                "terminal_pairs": TERMINAL_PAIRS,
                "pair_bin_edges_um": COUPLING_EDGES.tolist(),
                "ground_bin_edges_um": GROUND_EDGES.tolist(),
                "transform": "log1p of facing boundary length per absolute distance bin",
            },
            "shape_spectrum": {
                "offset": METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE,
                "dimensions": SHAPE_BLOCK_SIZE,
                "channels": [
                    "conductor two-point correlation",
                    "terminal cross-correlation",
                    "conductor width distribution",
                    "contour Fourier harmonics",
                ],
                "bin_edges_um": SHAPE_EDGES.tolist(),
            },
            "parameter_statistics": {
                "offset": METRIC_BLOCK_SIZE + COUPLING_BLOCK_SIZE + SHAPE_BLOCK_SIZE,
                "dimensions": PARAMETER_BLOCK_SIZE,
                "dimension_classes": list(PARAMETER_DIMENSION_CLASSES),
                "transform": "dimension-typed order statistics plus dimension-scoped signed hashing",
            },
            "physics_proxy": {
                "offset": V2_DIMENSIONS - PHYSICS_BLOCK_SIZE,
                "dimensions": PHYSICS_BLOCK_SIZE,
                "transform": "two-dimensional boundary-element capacitance proxy and dilation topology",
                "topology_radii_um": TOPOLOGY_RADII_UM.tolist(),
            },
        },
        "invariances": {
            "translation": "exact; every measurement is relative to the conductor set",
            "rotation": "approximate; correlation and harmonic magnitude blocks are orientation free",
            "scale": "deliberately absent; distances are absolute so capacitance scaling is learnable",
            "parameter_order": "sorted traversal and commutative accumulation",
            "parameter_schema": "fixed width for any parameter count or naming convention",
        },
    }
