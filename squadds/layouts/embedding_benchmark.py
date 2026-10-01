"""Controlled planar GDS fixtures for embedding acceptance experiments.

The generated files follow the same physical convention as the port-complete
SQuADDS sweeps: signal metal is on ``(1, 10)``, ground is on ``(1, 0)`` with
etch represented by one hole, and ordered terminal markers occupy consecutive
``(2 + terminal_index, 0)`` layers.  Two-terminal layouts therefore remain
byte-for-byte compatible with the published ``(2, 0)/(3, 0)`` convention while
the additive rule has no terminal-count ceiling.

These deliberately simple geometries complement, rather than replace, Qiskit
Metal components.  They isolate one physical variable at a time so failures in
an embedding can be attributed to a known perturbation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_SCHEMA_VERSION = "planar-embedding-acceptance-1.0"
CONDUCTOR_LAYER = (1, 10)
GROUND_LAYER = (1, 0)
PORT_LAYER_START = 2
GDS_DBU_UM = 0.001


def _require_geometry():
    try:
        import klayout.db as kdb
        import shapely
        from shapely import affinity
        from shapely.geometry import LineString, box
        from shapely.ops import nearest_points
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Embedding benchmark generation requires the SQuADDS gds extra.") from exc
    return kdb, shapely, affinity, LineString, box, nearest_points


def _polygon_parts(shape: Any) -> list[Any]:
    return [part for part in getattr(shape, "geoms", [shape]) if part.geom_type == "Polygon" and part.area > 0]


def _single_hole(shape: Any) -> Any:
    """Connect disjoint moat islands with negligible, deterministic bridges."""
    _, shapely, _, LineString, _, nearest_points = _require_geometry()
    hole = shape.buffer(0)
    while hole.geom_type == "MultiPolygon":
        parts = list(hole.geoms)
        candidates = []
        for first in range(len(parts)):
            for second in range(first + 1, len(parts)):
                candidates.append((float(parts[first].distance(parts[second])), first, second))
        if not candidates:
            break
        _, first, second = min(candidates)
        start, stop = nearest_points(parts[first], parts[second])
        # A sub-DBU bridge can disappear when written to integer GDS
        # coordinates and turn one intended moat into two holes.  A 4-nm-wide
        # bridge remains negligible at device scale but survives every pose.
        bridge = LineString([start, stop]).buffer(2.0 * GDS_DBU_UM, cap_style="square")
        hole = shapely.union_all([hole, bridge]).buffer(0)
    if hole.geom_type != "Polygon" or hole.is_empty or not hole.is_valid:
        raise ValueError("Benchmark moat could not be represented as one valid ground-plane hole.")
    return hole


def _radial_port(terminal: Any, center: np.ndarray, clearance_um: float, width_um: float) -> Any:
    """Build a port bridge from a terminal's outward support point to ground."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import nearest_points

    terminal_center = np.asarray(terminal.centroid.coords[0], dtype=float)
    direction = terminal_center - center
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1e-12:
        direction = np.asarray([1.0, 0.0])
    else:
        direction /= magnitude
    tangent = np.asarray([-direction[1], direction[0]])
    far = terminal_center + direction * (10.0 * max(math.sqrt(float(terminal.area)), clearance_um, 1.0))
    boundary_point, _ = nearest_points(terminal.boundary, Point(far))
    anchor = np.asarray(boundary_point.coords[0], dtype=float)
    half_width = 0.5 * min(width_um, max(math.sqrt(float(terminal.area)), GDS_DBU_UM))
    stop = anchor + direction * clearance_um
    return Polygon(
        [
            anchor - tangent * half_width,
            anchor + tangent * half_width,
            stop + tangent * half_width,
            stop - tangent * half_width,
        ]
    ).buffer(0)


def _insert_shape(cell: Any, layer_index: int, shape: Any, kdb: Any) -> None:
    scale = 1.0 / GDS_DBU_UM
    for polygon in _polygon_parts(shape):
        shell = [kdb.Point(int(round(x * scale)), int(round(y * scale))) for x, y in polygon.exterior.coords[:-1]]
        if len(shell) < 3:
            continue
        target = kdb.Polygon(shell)
        for ring in polygon.interiors:
            hole = [kdb.Point(int(round(x * scale)), int(round(y * scale))) for x, y in ring.coords[:-1]]
            if len(hole) >= 3:
                target.insert_hole(hole)
        cell.shapes(layer_index).insert(target)


def _write_layout(path: Path, terminals: Sequence[Any], ports: Sequence[Any], ground: Any) -> None:
    kdb, _, _, _, _, _ = _require_geometry()
    layout = kdb.Layout()
    layout.dbu = GDS_DBU_UM
    cell = layout.create_cell("TOP")
    conductor_layer = layout.layer(*CONDUCTOR_LAYER)
    ground_layer = layout.layer(*GROUND_LAYER)
    for terminal in terminals:
        _insert_shape(cell, conductor_layer, terminal, kdb)
    _insert_shape(cell, ground_layer, ground, kdb)
    for terminal_index, port in enumerate(ports):
        _insert_shape(cell, layout.layer(PORT_LAYER_START + terminal_index, 0), port, kdb)
    options = kdb.SaveLayoutOptions()
    options.gds2_write_timestamps = False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.gds")
    layout.write(str(temporary), options)
    temporary.replace(path)


def write_standard_planar_gds(
    path: str | Path,
    terminals: Sequence[Any],
    *,
    clearance_um: float = 4.0,
    domain_padding_um: float = 80.0,
    port_width_um: float = 2.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one arbitrary-terminal planar layout and its semantic sidecar."""
    _, shapely, _, _, box, _ = _require_geometry()
    path = Path(path)
    terminals = [terminal.buffer(0) for terminal in terminals]
    if len(terminals) < 2 or any(shape.is_empty or not shape.is_valid or shape.area <= 0 for shape in terminals):
        raise ValueError("A benchmark layout requires at least two valid terminal polygons.")
    conductor = shapely.union_all(terminals)
    center = np.asarray(conductor.centroid.coords[0], dtype=float)
    ports = [_radial_port(shape, center, clearance_um, port_width_um) for shape in terminals]
    moat = _single_hole(shapely.union_all([conductor.buffer(clearance_um, join_style="mitre"), *ports]))
    left, bottom, right, top = moat.bounds
    outer = box(
        left - domain_padding_um,
        bottom - domain_padding_um,
        right + domain_padding_um,
        top + domain_padding_um,
    )
    ground = outer.difference(moat)
    _write_layout(path, terminals, ports, ground)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "gds_file": path.name,
        "gds_sha256": digest,
        "units": "um",
        "terminal_count": len(terminals),
        "layers": [
            {
                "layer": 1,
                "datatype": 0,
                "role": "ground",
                "plane_id": "device",
                "material": "perfect_conductor",
            },
            {
                "layer": 1,
                "datatype": 10,
                "role": "conductor",
                "plane_id": "device",
                "material": "perfect_conductor",
            },
            *[
                {
                    "layer": PORT_LAYER_START + index,
                    "datatype": 0,
                    "role": "port",
                    "plane_id": "device",
                    "terminal_index": index,
                    "terminal_id": f"t{index}",
                    "net_id": f"t{index}",
                    "conductor_area_um2": float(terminals[index].area),
                    "conductor_perimeter_um": float(terminals[index].length),
                }
                for index in range(len(terminals))
            ],
        ],
        "planes": [
            {
                "plane_id": "device",
                "z_um": 0.0,
                "conductor_layers": [[1, 10]],
                "ground_layers": [[1, 0]],
                "material": "perfect_conductor",
                "thickness_um": None,
            }
        ],
        "materials": [],
        "future_stack_hook": {
            "implemented_geometry": "planar",
            "plane": "device",
            "z_um": 0.0,
            "material_stack_file": None,
        },
        **dict(metadata or {}),
    }
    sidecar_path = path.with_suffix(".layers.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar


def rectangular_pair(gap_um: float = 6.0, facing_length_um: float = 30.0, width_um: float = 36.0) -> list[Any]:
    """Return two facing rectangular terminals centered at the origin."""
    _, _, _, _, box, _ = _require_geometry()
    half_gap = 0.5 * gap_um
    half_height = 0.5 * facing_length_um
    return [
        box(-half_gap - width_um, -half_height, -half_gap, half_height),
        box(half_gap, -half_height, half_gap + width_um, half_height),
    ]


def radial_terminals(count: int, radius_um: float = 55.0, pad_size_um: float = 18.0) -> list[Any]:
    """Return ``count`` equal pads on a circle for terminal-count controls."""
    _, _, affinity, _, box, _ = _require_geometry()
    if count < 2:
        raise ValueError("count must be at least two")
    base = box(-0.5 * pad_size_um, -0.5 * pad_size_um, 0.5 * pad_size_um, 0.5 * pad_size_um)
    terminals = []
    for index in range(count):
        angle = 360.0 * index / count
        shifted = affinity.translate(base, xoff=radius_um, yoff=0.0)
        terminals.append(affinity.rotate(shifted, angle, origin=(0.0, 0.0)))
    return terminals


def _transform_shapes(shapes: Iterable[Any], *, angle: float = 0.0, reflect: bool = False, scale: float = 1.0, shift=(0.0, 0.0)) -> list[Any]:
    _, _, affinity, _, _, _ = _require_geometry()
    answer = []
    for shape in shapes:
        transformed = affinity.scale(shape, xfact=-1.0 if reflect else 1.0, yfact=1.0, origin=(0.0, 0.0))
        transformed = affinity.rotate(transformed, angle, origin=(0.0, 0.0))
        transformed = affinity.scale(transformed, xfact=scale, yfact=scale, origin=(0.0, 0.0))
        transformed = affinity.translate(transformed, xoff=float(shift[0]), yoff=float(shift[1]))
        answer.append(transformed)
    return answer


def generate_controlled_planar_suite(output_dir: str | Path) -> list[dict[str, Any]]:
    """Generate the pre-registered nuisance, physical, and terminal-count suite."""
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []

    def emit(name: str, terminals: Sequence[Any], group: str, value: Any, **options: Any) -> None:
        path = output_dir / f"{name}.gds"
        sidecar = write_standard_planar_gds(
            path,
            terminals,
            metadata={"fixture_name": name, "perturbation_group": group, "control_value": value},
            **options,
        )
        rows.append(
            {
                "fixture_name": name,
                "gds_path": str(path),
                "sidecar_path": str(path.with_suffix(".layers.json")),
                "perturbation_group": group,
                "control_value": value,
                "terminal_count": sidecar["terminal_count"],
                "gds_sha256": sidecar["gds_sha256"],
            }
        )

    baseline = rectangular_pair()
    emit("pair_baseline", baseline, "nuisance", "baseline")
    emit("pair_translated", _transform_shapes(baseline, shift=(4312.0, -2789.0)), "nuisance", "translation")
    emit("pair_rotated_90", _transform_shapes(baseline, angle=90.0), "nuisance", "rotation_90")
    emit("pair_rotated_37", _transform_shapes(baseline, angle=37.0), "nuisance", "rotation_37")
    emit("pair_reflected", _transform_shapes(baseline, reflect=True), "nuisance", "reflection")
    emit("pair_ground_crop_small", baseline, "nuisance", "ground_crop_small", domain_padding_um=50.0)
    emit("pair_ground_crop_large", baseline, "nuisance", "ground_crop_large", domain_padding_um=500.0)

    for factor in (0.5, 0.7, 1.0, 1.4, 2.0):
        emit(
            f"pair_scale_{str(factor).replace('.', 'p')}",
            _transform_shapes(baseline, scale=factor),
            "uniform_scale",
            factor,
            clearance_um=4.0 * factor,
            port_width_um=2.0 * factor,
            domain_padding_um=80.0 * factor,
        )
    for gap in (1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 14.0, 22.0):
        emit(f"pair_gap_{str(gap).replace('.', 'p')}", rectangular_pair(gap_um=gap), "terminal_gap_um", gap)
    for length in (8.0, 12.0, 18.0, 27.0, 40.0, 60.0, 90.0):
        emit(
            f"pair_facing_{str(length).replace('.', 'p')}",
            rectangular_pair(facing_length_um=length),
            "facing_length_um",
            length,
        )
    for clearance in (2.0, 3.0, 4.0, 6.0, 9.0, 14.0):
        emit(
            f"pair_ground_gap_{str(clearance).replace('.', 'p')}",
            baseline,
            "ground_clearance_um",
            clearance,
            clearance_um=clearance,
        )
    for count in (2, 3, 4, 5, 6, 8, 12):
        emit(f"radial_{count}_terminal", radial_terminals(count), "terminal_count", count)

    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "rows": rows,
        "layer_contract": {
            "ground": list(GROUND_LAYER),
            "conductor": list(CONDUCTOR_LAYER),
            "ordered_ports": "(2 + terminal_index, 0)",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return rows


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "CONDUCTOR_LAYER",
    "GROUND_LAYER",
    "PORT_LAYER_START",
    "generate_controlled_planar_suite",
    "radial_terminals",
    "rectangular_pair",
    "write_standard_planar_gds",
]
