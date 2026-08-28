"""Faithful Qiskit Metal GDS export with explicit two-terminal port markers.

Qiskit Metal stores component geometry in millimetres.  SQuADDS layout GDS
files use micrometres and reserve ``(1, 10)`` for signal metal, ``(1, 11)``
for subtractive etch, and ``(2, 0)`` / ``(3, 0)`` for the ordered terminals.
The helpers here keep those concerns in one place so generated datasets and
their validation use exactly the same geometry contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MM_TO_UM = 1000.0
PORT_LENGTH_UM = 2.0


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


def transmon_cross_port_markers(component: Any, design: Any) -> list[PortMarker]:
    """Return cross/junction then readout/claw markers for ``TransmonCross``."""
    junctions = design.qgeometry.tables["junction"]
    rows = junctions[(junctions.component == component.id) & (junctions.name == "rect_jj")]
    if len(rows) != 1:
        raise ValueError(f"Expected one rect_jj junction for {component.name!r}; found {len(rows)}.")
    row = rows.iloc[0]
    cross = marker_from_junction(
        row.geometry,
        qgeometry_role(design, subtract=False),
        width_mm=float(row.width),
        semantic="cross_junction_port",
        layer=2,
        source="junction:rect_jj",
    )
    readout = marker_from_pin(
        component.pins["readout"],
        semantic="readout_claw_port",
        layer=3,
        source="pin:readout",
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
        ),
        marker_from_pin(
            component.pins["second_end"],
            semantic="second_bottom_port",
            layer=3,
            source="pin:second_end",
        ),
    ]


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


def _chip_ground_geometry(design: Any, etch: Any) -> Any:
    """Return the planar main-chip ground after subtractive qgeometry is cut out."""
    from shapely.geometry import box

    size = design.parse_value(design.chips.main.size)
    center_x = float(size["center_x"])
    center_y = float(size["center_y"])
    half_x = float(size["size_x"]) / 2
    half_y = float(size["size_y"]) / 2
    domain = box(center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y)
    return domain.difference(etch)


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
) -> None:
    """Write Metal qgeometry and optional ordered terminal markers atomically.

    When requested, the ground is the configured Qiskit Metal main-chip box
    minus all subtractive qgeometry.  It is kept on ``(1, 0)`` and never mixed
    with the explicit signal and etch roles.
    """
    kdb = _require_klayout()
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    for table_name in ("poly", "path"):
        table = design.qgeometry.tables[table_name]
        for row in table.itertuples(index=False):
            if row.helper:
                continue
            geometry = row.geometry
            if table_name == "path":
                geometry = geometry.buffer(float(row.width) / 2, cap_style="flat", join_style="round")
            layer_index = layout.layer(int(row.layer), 11 if row.subtract else 10)
            for polygon in each_polygon(geometry):
                _insert_polygon(top, layer_index, polygon, kdb)
    if include_ground_domain:
        domain_layer = layout.layer(1, 0)
        domain = _chip_ground_geometry(design, qgeometry_role(design, subtract=True))
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


def validate_ported_gds(design: Any, path: Path, markers: Iterable[PortMarker]) -> dict[str, Any]:
    """Validate qgeometry fidelity and unambiguous two-terminal port assignment."""
    from shapely.affinity import scale

    from squadds.layouts.geometry_v2 import read_layer_geometry

    markers = list(markers)
    geometry = read_layer_geometry(path)
    checks: dict[str, bool] = {}
    role_metrics: dict[str, dict[str, float]] = {}
    for name, key, subtract in (
        ("conductor", (1, 10), False),
        ("etch", (1, 11), True),
    ):
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

    checks["two_markers"] = len(markers) == 2 and all((marker.layer, marker.datatype) in geometry for marker in markers)
    expected_layers = [(2, 0), (3, 0)]
    checks["ordered_layers"] = [(marker.layer, marker.datatype) for marker in markers] == expected_layers

    conductor = geometry.get((1, 10))
    assignments: list[int] = []
    distances: list[float] = []
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
            distances.append(float(min(part_distances)))
    checks["two_signal_terminals"] = component_count == 2
    checks["unique_terminal_assignment"] = len(assignments) == 2 and len(set(assignments)) == 2
    checks["markers_touch_signal"] = len(distances) == 2 and max(distances) <= 0.001

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
        "port_distances_um": distances,
    }
