"""Faithful Qiskit Metal GDS export with explicit two-terminal port markers.

Qiskit Metal stores component geometry in millimetres.  SQuADDS layout GDS
files use micrometres and reserve ``(1, 0)`` for a ground plane with one etched
hole, ``(1, 10)`` for signal metal, and ``(2, 0)`` / ``(3, 0)`` for the ordered
terminals.  The helpers here keep those concerns in one place so generated
datasets and their validation use exactly the same geometry contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MM_TO_UM = 1000.0
GROUND_DOMAIN_SCALE = 5.8
PORT_LENGTH_UM = 2.0
_BRIDGE_HALF_WIDTH_MM = 1e-6


@dataclass(frozen=True)
class PortMarker:
    """One ordered GDS terminal marker expressed in Qiskit Metal millimetres."""

    semantic: str
    layer: int
    polygon: Any
    source: str
    datatype: int = 0


def _require_klayout():
    try:
        import klayout.db as kdb
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("GDS generation requires `uv sync --extra gds`.") from exc
    return kdb


def each_polygon(geometry: Any):
    """Yield polygonal leaves from a Shapely geometry."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from each_polygon(child)


def qgeometry_role(design: Any, *, subtract: bool) -> Any:
    """Return the union of rendered poly/path qgeometry in Metal's mm units."""
    from shapely import union_all

    geometries = []
    for table_name in ("poly", "path"):
        table = design.qgeometry.tables[table_name]
        for row in table.itertuples(index=False):
            if bool(row.helper) or bool(row.subtract) != subtract:
                continue
            geometry = row.geometry
            if table_name == "path":
                geometry = geometry.buffer(float(row.width) / 2, cap_style="flat", join_style="round")
            geometries.append(geometry)
    return union_all(geometries)


def marker_from_pin(
    pin: Any,
    *,
    semantic: str,
    layer: int,
    source: str,
    length_um: float = PORT_LENGTH_UM,
) -> PortMarker:
    """Extend a Metal pin cross-section outward by ``length_um``."""
    from shapely.geometry import Polygon

    points = np.asarray(pin["points"], dtype=float)
    normal = np.asarray(pin["normal"], dtype=float)
    if points.shape != (2, 2):
        raise ValueError(f"Port source {source!r} does not have a two-point pin cross-section.")
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 0:
        raise ValueError(f"Port source {source!r} has a zero normal.")
    extension = normal / magnitude * (float(length_um) / MM_TO_UM)
    polygon = Polygon([points[0], points[1], points[1] + extension, points[0] + extension])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"Port source {source!r} produced an invalid marker.")
    return PortMarker(semantic=semantic, layer=layer, polygon=polygon, source=source)


def marker_from_junction(
    junction: Any,
    conductor: Any,
    *,
    width_mm: float,
    semantic: str,
    layer: int,
    source: str,
    length_um: float = PORT_LENGTH_UM,
) -> PortMarker:
    """Place a marker outside the conductor at the connected junction endpoint."""
    from shapely.geometry import Point, Polygon

    coordinates = np.asarray(junction.coords, dtype=float)
    if len(coordinates) != 2:
        raise ValueError(f"Junction source {source!r} must be a two-point line.")
    distances = [float(conductor.distance(Point(point))) for point in coordinates]
    connected_index = int(np.argmin(distances))
    connected = coordinates[connected_index]
    outside = coordinates[1 - connected_index]
    direction = outside - connected
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 0:
        raise ValueError(f"Junction source {source!r} has zero length.")
    normal = direction / magnitude
    tangent = np.array([-normal[1], normal[0]])
    half_width = float(width_mm) / 2
    extension = normal * (float(length_um) / MM_TO_UM)
    polygon = Polygon(
        [
            connected - tangent * half_width,
            connected + tangent * half_width,
            connected + tangent * half_width + extension,
            connected - tangent * half_width + extension,
        ]
    )
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"Junction source {source!r} produced an invalid marker.")
    return PortMarker(semantic=semantic, layer=layer, polygon=polygon, source=source)


def _parsed_length_um(component: Any, value: Any) -> float:
    """Parse one Qiskit Metal length option and return micrometres."""
    length_um = float(component.design.parse_value(value)) * MM_TO_UM
    if length_um <= 0:
        raise ValueError(f"Expected a positive component clearance; got {value!r}.")
    return length_um


def transmon_cross_port_markers(component: Any, design: Any) -> list[PortMarker]:
    """Return native-clearance cross and claw bridges to ground."""
    junctions = design.qgeometry.tables["junction"]
    rows = junctions[(junctions.component == component.id) & (junctions.name == "rect_jj")]
    if len(rows) != 1:
        raise ValueError(f"Expected one rect_jj junction for {component.name!r}; found {len(rows)}.")
    row = rows.iloc[0]
    cross_clearance_um = _parsed_length_um(component, component.options.cross_gap)
    claw_clearance_um = _parsed_length_um(
        component,
        component.options.connection_pads["readout"]["claw_gap"],
    )
    cross = marker_from_junction(
        row.geometry,
        qgeometry_role(design, subtract=False),
        width_mm=float(row.width),
        semantic="cross_junction_port",
        layer=2,
        source="junction:rect_jj",
        length_um=cross_clearance_um,
    )
    readout = marker_from_pin(
        component.pins["readout"],
        semantic="readout_claw_port",
        layer=3,
        source="pin:readout",
        length_um=claw_clearance_um,
    )
    return [cross, readout]


def capn_interdigital_tee_port_markers(component: Any) -> list[PortMarker]:
    """Return one marker for each electrical conductor in the legacy CapN tee.

    ``prime_start`` and ``prime_end`` are two routing ends of the same prime/top
    conductor.  Only ``prime_start`` is marked so the file remains a two-terminal
    capacitance layout; ``second_end`` marks the other (second/bottom) conductor.
    """
    return [
        marker_from_pin(
            component.pins["prime_start"],
            semantic="prime_top_port",
            layer=2,
            source="pin:prime_start",
            length_um=_parsed_length_um(component, component.options.prime_gap),
        ),
        marker_from_pin(
            component.pins["second_end"],
            semantic="second_bottom_port",
            layer=3,
            source="pin:second_end",
            length_um=_parsed_length_um(component, component.options.second_gap),
        ),
    ]


def minimum_ground_clearance_um(component: Any) -> float:
    """Return the smallest native signal-to-ground clearance for a component."""
    if component.__class__.__name__ == "TransmonCross":
        return min(
            _parsed_length_um(component, component.options.cross_gap),
            _parsed_length_um(component, component.options.connection_pads["readout"]["claw_gap"]),
        )
    if component.__class__.__name__ == "CapNInterdigitalTee":
        clearance = component.options.get("cap_gap_ground")
        if not isinstance(clearance, (str, int, float)):
            clearance = min(
                component.design.parse_value(component.options.prime_gap),
                component.design.parse_value(component.options.second_gap),
            )
        return _parsed_length_um(component, clearance)
    raise ValueError(f"No standardized ground clearance for {component.__class__.__name__!r}.")


def _insert_polygon(cell: Any, layer_index: int, polygon: Any, kdb: Any) -> None:
    shell = [kdb.DPoint(float(x) * MM_TO_UM, float(y) * MM_TO_UM) for x, y in polygon.exterior.coords[:-1]]
    if len(shell) < 3:
        return
    target = kdb.DPolygon(shell)
    for interior in polygon.interiors:
        hole = [kdb.DPoint(float(x) * MM_TO_UM, float(y) * MM_TO_UM) for x, y in interior.coords[:-1]]
        if len(hole) >= 3:
            target.insert_hole(hole)
    cell.shapes(layer_index).insert(target)


def insert_port_markers(layout: Any, cell: Any, markers: Iterable[PortMarker]) -> None:
    """Insert markers into an existing KLayout layout/cell."""
    kdb = _require_klayout()
    for marker in markers:
        layer_index = layout.layer(int(marker.layer), int(marker.datatype))
        for polygon in each_polygon(marker.polygon):
            _insert_polygon(cell, layer_index, polygon, kdb)


def _single_connected_hole(geometry: Any) -> Any:
    """Join disjoint QMetal etch polygons with grid-scale, minimum-length bridges."""
    from shapely import union_all
    from shapely.geometry import LineString
    from shapely.ops import nearest_points

    hole = geometry.buffer(0)
    while hole.geom_type == "MultiPolygon":
        parts = list(hole.geoms)
        best: tuple[float, int, int] | None = None
        for first in range(len(parts)):
            for second in range(first + 1, len(parts)):
                candidate = (float(parts[first].distance(parts[second])), first, second)
                if best is None or candidate < best:
                    best = candidate
        if best is None:  # pragma: no cover - an empty MultiPolygon is not produced here
            break
        _, first, second = best
        start, end = nearest_points(parts[first], parts[second])
        bridge = LineString([start, end]).buffer(
            _BRIDGE_HALF_WIDTH_MM,
            cap_style="square",
            join_style="mitre",
        )
        hole = union_all([hole, bridge]).buffer(0)
    if hole.geom_type != "Polygon" or hole.is_empty or not hole.is_valid:
        raise ValueError("The standardized subtractive geometry did not form one valid hole.")
    return hole


def standardized_ground_geometry(
    design: Any,
    markers: Iterable[PortMarker],
    *,
    minimum_clearance_um: float,
    domain_scale: float = GROUND_DOMAIN_SCALE,
) -> tuple[Any, Any]:
    """Return a centered dynamic ground and its single QMetal-derived hole.

    The native subtractive qgeometry remains authoritative.  A minimum buffer
    closes the flat CPW ends emitted by Qiskit Metal, terminal bridge polygons
    make the two lumped ports reach the ground boundary, and only disjoint etch
    islands receive negligible-width bridges so the ground has one hole.
    """
    from shapely import union_all
    from shapely.geometry import box

    conductor = qgeometry_role(design, subtract=False)
    clearance_mm = float(minimum_clearance_um) / MM_TO_UM
    if conductor.is_empty or clearance_mm <= 0:
        raise ValueError("Standardized ground generation requires conductor geometry and positive clearance.")
    native_etch = qgeometry_role(design, subtract=True)
    marker_shapes = [marker.polygon for marker in markers]
    hole = _single_connected_hole(
        union_all(
            [
                native_etch,
                conductor.buffer(clearance_mm, join_style="mitre"),
                *marker_shapes,
            ]
        )
    )

    left, bottom, right, top = conductor.bounds
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (bottom + top)
    side = float(domain_scale) * max(right - left, top - bottom)
    if side <= 0:
        raise ValueError("Cannot size a ground plane around zero-span conductor geometry.")
    half_side = side / 2
    domain = box(center_x - half_side, center_y - half_side, center_x + half_side, center_y + half_side)
    if not domain.contains(hole):
        raise ValueError("The dynamic ground domain does not contain the complete etch hole.")
    return domain.difference(hole), hole


def _write_without_timestamps(layout: Any, path: Path, kdb: Any) -> None:
    """Write deterministic GDS bytes by omitting volatile creation timestamps."""
    options = kdb.SaveLayoutOptions()
    options.gds2_write_timestamps = False
    layout.write(str(path), options)


def export_qgeometry_gds(
    design: Any,
    destination: Path,
    *,
    markers: Iterable[PortMarker] = (),
    include_ground_domain: bool = False,
    minimum_ground_clearance_um: float | None = None,
) -> None:
    """Write Metal qgeometry and optional ordered terminal markers atomically.

    When requested, the ground follows the GeneralizedCapNInterdigital
    convention: a dynamic square domain on ``(1, 0)`` with one subtractive hole
    and no filled ``(1, 11)`` etch layer.
    """
    kdb = _require_klayout()
    markers = list(markers)
    if include_ground_domain and minimum_ground_clearance_um is None:
        raise ValueError("Ground-domain export requires minimum_ground_clearance_um.")
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    for table_name in ("poly", "path"):
        table = design.qgeometry.tables[table_name]
        for row in table.itertuples(index=False):
            if row.helper:
                continue
            if include_ground_domain and row.subtract:
                continue
            geometry = row.geometry
            if table_name == "path":
                geometry = geometry.buffer(float(row.width) / 2, cap_style="flat", join_style="round")
            layer_index = layout.layer(int(row.layer), 11 if row.subtract else 10)
            for polygon in each_polygon(geometry):
                _insert_polygon(top, layer_index, polygon, kdb)
    if include_ground_domain:
        domain_layer = layout.layer(1, 0)
        domain, _ = standardized_ground_geometry(
            design,
            markers,
            minimum_clearance_um=float(minimum_ground_clearance_um),
        )
        for polygon in each_polygon(domain):
            _insert_polygon(top, domain_layer, polygon, kdb)
    insert_port_markers(layout, top, markers)
    temporary = destination.with_suffix(".tmp.gds")
    _write_without_timestamps(layout, temporary, kdb)
    temporary.replace(destination)


def add_port_markers_to_gds(source: Path, destination: Path, markers: Iterable[PortMarker]) -> None:
    """Copy a rendered GDS while adding markers, using an atomic destination."""
    kdb = _require_klayout()
    layout = kdb.Layout()
    layout.read(str(source))
    top = layout.top_cell()
    if top is None:
        raise ValueError(f"GDS file has no top cell: {source}")
    insert_port_markers(layout, top, markers)
    temporary = destination.with_suffix(".tmp.gds")
    _write_without_timestamps(layout, temporary, kdb)
    temporary.replace(destination)


def validate_ported_gds(
    design: Any,
    path: Path,
    markers: Iterable[PortMarker],
    *,
    minimum_ground_clearance_um: float,
) -> dict[str, Any]:
    """Validate the unified layer, ground-hole, and two-terminal contract."""
    from shapely.affinity import scale
    from shapely.geometry import Polygon

    from squadds.layouts.geometry_v2 import read_layer_geometry

    markers = list(markers)
    geometry = read_layer_geometry(path)
    checks: dict[str, bool] = {}
    role_metrics: dict[str, dict[str, float]] = {}
    for name, key, subtract in (("conductor", (1, 10), False),):
        expected_mm = qgeometry_role(design, subtract=subtract)
        expected = scale(expected_mm, xfact=MM_TO_UM, yfact=MM_TO_UM, origin=(0, 0))
        actual = geometry.get(key)
        if actual is None:
            role_metrics[name] = {"expected_area_um2": float(expected.area), "actual_area_um2": 0.0, "xor_area_um2": float(expected.area)}
            checks[f"{name}_roundtrip"] = False
            continue
        xor_area = float(expected.symmetric_difference(actual).area)
        tolerance = max(0.01, float(expected.area) * 1e-5)
        role_metrics[name] = {
            "expected_area_um2": float(expected.area),
            "actual_area_um2": float(actual.area),
            "xor_area_um2": xor_area,
            "tolerance_um2": tolerance,
        }
        checks[f"{name}_roundtrip"] = xor_area <= tolerance

    expected_ground_mm, _ = standardized_ground_geometry(
        design,
        markers,
        minimum_clearance_um=minimum_ground_clearance_um,
    )
    expected_ground = scale(expected_ground_mm, xfact=MM_TO_UM, yfact=MM_TO_UM, origin=(0, 0))
    ground = geometry.get((1, 0))
    ground_xor = (
        float(expected_ground.symmetric_difference(ground).area)
        if ground is not None
        else float(expected_ground.area)
    )
    ground_tolerance = max(0.1, float(expected_ground.area) * 1e-7)
    role_metrics["ground"] = {
        "expected_area_um2": float(expected_ground.area),
        "actual_area_um2": float(ground.area) if ground is not None else 0.0,
        "xor_area_um2": ground_xor,
        "tolerance_um2": ground_tolerance,
    }
    checks["ground_roundtrip"] = ground_xor <= ground_tolerance
    checks["exact_layer_set"] = set(geometry) == {(1, 0), (1, 10), (2, 0), (3, 0)}
    checks["etch_layer_absent"] = (1, 11) not in geometry

    conductor = geometry.get((1, 10))
    ground_polygons = list(getattr(ground, "geoms", [ground])) if ground is not None else []
    ground_holes = [interior for polygon in ground_polygons for interior in getattr(polygon, "interiors", [])]
    checks["one_ground_polygon"] = len(ground_polygons) == 1
    checks["one_ground_hole"] = len(ground_holes) == 1
    actual_hole = Polygon(ground_holes[0]) if len(ground_holes) == 1 else None
    checks["hole_contains_conductor"] = (
        conductor is not None and actual_hole is not None and bool(actual_hole.covers(conductor))
    )

    if ground is not None and conductor is not None:
        ground_width = float(ground.bounds[2] - ground.bounds[0])
        ground_height = float(ground.bounds[3] - ground.bounds[1])
        conductor_width = float(conductor.bounds[2] - conductor.bounds[0])
        conductor_height = float(conductor.bounds[3] - conductor.bounds[1])
        ground_aspect = ground_width / ground_height
        ground_scale = max(ground_width, ground_height) / max(conductor_width, conductor_height)
        ground_center_offset = [
            0.5 * (ground.bounds[0] + ground.bounds[2] - conductor.bounds[0] - conductor.bounds[2]),
            0.5 * (ground.bounds[1] + ground.bounds[3] - conductor.bounds[1] - conductor.bounds[3]),
        ]
    else:
        ground_aspect = ground_scale = float("nan")
        ground_center_offset = [float("nan"), float("nan")]
    checks["ground_aspect"] = 0.9 <= ground_aspect <= 2.0
    checks["ground_scale"] = 4.2 <= ground_scale <= 8.0
    checks["ground_centered"] = max(abs(value) for value in ground_center_offset) <= 0.001

    checks["two_markers"] = len(markers) == 2 and all((marker.layer, marker.datatype) in geometry for marker in markers)
    expected_layers = [(2, 0), (3, 0)]
    checks["ordered_layers"] = [(marker.layer, marker.datatype) for marker in markers] == expected_layers

    assignments: list[int] = []
    conductor_distances: list[float] = []
    ground_distances: list[float] = []
    component_count = 0
    if conductor is not None:
        parts = [part for part in getattr(conductor, "geoms", [conductor]) if part.area > 0]
        component_count = len(parts)
        for marker in markers:
            port = geometry.get((marker.layer, marker.datatype))
            if port is None or not parts:
                continue
            part_distances = [float(part.distance(port)) for part in parts]
            assignments.append(int(np.argmin(part_distances)))
            conductor_distances.append(float(min(part_distances)))
            ground_distances.append(float(ground.distance(port)) if ground is not None else float("inf"))
    checks["two_signal_terminals"] = component_count == 2
    checks["unique_terminal_assignment"] = len(assignments) == 2 and len(set(assignments)) == 2
    checks["markers_touch_signal"] = len(conductor_distances) == 2 and max(conductor_distances) <= 0.001
    checks["markers_touch_ground"] = len(ground_distances) == 2 and max(ground_distances) <= 0.001

    port_metrics = []
    for marker in markers:
        expected = scale(marker.polygon, xfact=MM_TO_UM, yfact=MM_TO_UM, origin=(0, 0))
        actual = geometry.get((marker.layer, marker.datatype))
        xor = float(expected.symmetric_difference(actual).area) if actual is not None else float(expected.area)
        port_metrics.append(
            {
                "semantic": marker.semantic,
                "source": marker.source,
                "layer": marker.layer,
                "datatype": marker.datatype,
                "area_um2": float(expected.area),
                "xor_area_um2": xor,
            }
        )
    checks["marker_roundtrip"] = all(item["xor_area_um2"] <= 0.01 for item in port_metrics)
    return {
        "path": str(path),
        "valid": all(checks.values()),
        "checks": checks,
        "qgeometry": role_metrics,
        "ports": port_metrics,
        "signal_component_count": component_count,
        "port_component_assignments": assignments,
        "port_conductor_distances_um": conductor_distances,
        "port_ground_distances_um": ground_distances,
        "ground_aspect": ground_aspect,
        "ground_scale": ground_scale,
        "ground_center_offset_um": ground_center_offset,
        "ground_hole_count": len(ground_holes),
    }
