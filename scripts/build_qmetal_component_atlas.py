#!/usr/bin/env python
"""Build an auditable singleton atlas of Qiskit Metal's planar components.

The inventory is intentionally frozen rather than discovered at run time.  A
new upstream component must therefore be reviewed before it enters a published
atlas.  Geometry-equivalent classes retain the same equivalence identifier;
the generator never adds class-dependent geometry merely to separate labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
from shapely.wkb import dumps as wkb_dumps

ATLAS_SCHEMA_VERSION = "1.0.0"
INVENTORY_VERSION = "qiskit-metal-0.1.2-singletons-v1"
GROUND_PADDING_UM = 169.0
DEFAULT_CLEARANCE_UM = 6.0
MM_TO_UM = 1000.0


@dataclass(frozen=True)
class ComponentSpec:
    category: str
    class_name: str
    import_path: str
    recipe: str = "default"


def _specs(category: str, module_and_names: list[tuple[str, str]], recipe: str = "default"):
    return tuple(
        ComponentSpec(category, name, f"qiskit_metal.qlibrary.{module}.{name}", recipe)
        for module, name in module_and_names
    )


COMPONENT_SPECS = (
    *_specs(
        "couplers",
        [
            ("couplers.cap_n_interdigital_tee", "CapNInterdigitalTee"),
            ("couplers.coupled_line_tee", "CoupledLineTee"),
            ("couplers.line_tee", "LineTee"),
            ("couplers.tunable_coupler_01", "TunableCoupler01"),
        ],
    ),
    *_specs(
        "lumped",
        [
            ("lumped.cap_3_interdigital", "Cap3Interdigital"),
            ("lumped.cap_n_interdigital", "CapNInterdigital"),
            ("lumped.resonator_coil_rect", "ResonatorCoilRect"),
        ],
    ),
    *_specs(
        "qubits",
        [
            ("qubits.JJ_Dolan", "jj_dolan"),
            ("qubits.JJ_Manhattan", "jj_manhattan"),
            ("qubits.SQUID_loop", "SQUID_LOOP"),
            ("qubits.Transmon_Interdigitated", "TransmonInterdigitated"),
            ("qubits.star_qubit", "StarQubit"),
            ("qubits.transmon_concentric", "TransmonConcentric"),
            ("qubits.transmon_cross", "TransmonCross"),
            ("qubits.transmon_cross_fl", "TransmonCrossFL"),
            ("qubits.transmon_pocket", "TransmonPocket"),
            ("qubits.transmon_pocket_6", "TransmonPocket6"),
            ("qubits.transmon_pocket_cl", "TransmonPocketCL"),
            ("qubits.transmon_pocket_teeth", "TransmonPocketTeeth"),
        ],
    ),
    *_specs("resonator", [("resonator.readoutres_fc", "ReadoutResFC")]),
    *_specs(
        "sample_shapes",
        [
            ("sample_shapes.circle_caterpillar", "CircleCaterpillar"),
            ("sample_shapes.circle_raster", "CircleRaster"),
            ("sample_shapes.n_gon", "NGon"),
            ("sample_shapes.n_square_spiral", "NSquareSpiral"),
            ("sample_shapes.rectangle", "Rectangle"),
            ("sample_shapes.rectangle_hollow", "RectangleHollow"),
            ("sample_shapes.smiley_face", "SmileyFace"),
        ],
    ),
    *_specs(
        "terminations",
        [
            ("terminations.launchpad_wb", "LaunchpadWirebond"),
            ("terminations.launchpad_wb_coupled", "LaunchpadWirebondCoupled"),
            ("terminations.launchpad_wb_driven", "LaunchpadWirebondDriven"),
            ("terminations.open_to_ground", "OpenToGround"),
            ("terminations.short_to_ground", "ShortToGround"),
        ],
    ),
    *_specs(
        "routes",
        [
            ("tlines.anchored_path", "RouteAnchors"),
            ("tlines.framed_path", "RouteFramed"),
            ("tlines.meandered", "RouteMeander"),
            ("tlines.mixed_path", "RouteMixed"),
            ("tlines.pathfinder", "RoutePathfinder"),
            ("tlines.straight_path", "RouteStraight"),
        ],
        recipe="contextual_route",
    ),
)


def _component_class(spec: ComponentSpec):
    module_name, _, class_name = spec.import_path.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def _source_sha256(component_class: type) -> str | None:
    module = importlib.import_module(component_class.__module__)
    source = getattr(module, "__file__", None)
    if not source:
        return None
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _git_metadata(component_class: type) -> dict[str, Any]:
    module = importlib.import_module(component_class.__module__)
    source = Path(module.__file__).resolve()
    root = None
    for parent in source.parents:
        if (parent / ".git").exists():
            root = parent
            break
    result: dict[str, Any] = {"source_root": str(root or source.parents[2])}
    if root is None:
        return result

    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        result["commit"] = run("rev-parse", "HEAD")
        result["describe"] = run("describe", "--tags", "--always", "--dirty")
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        result["dirty_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        result["dirty"] = bool(diff)
    except (OSError, subprocess.CalledProcessError) as error:
        result["git_error"] = f"{type(error).__name__}: {error}"
    return result


def _route_options(class_name: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "pin_inputs": {
            "start_pin": {"component": "atlas_start", "pin": "tie"},
            "end_pin": {"component": "atlas_end", "pin": "tie"},
        },
        "total_length": "5mm",
    }
    if class_name in {"RouteAnchors", "RouteMixed", "RoutePathfinder"}:
        options["anchors"] = {0: (-0.5, 0.5), 1: (0.5, -0.5)}
    return options


def instantiate(spec: ComponentSpec):
    """Return a fresh design and target, adding fixtures only when required."""
    from qiskit_metal import designs

    design = designs.DesignPlanar()
    design.overwrite_enabled = True
    component_class = _component_class(spec)
    if spec.recipe == "contextual_route":
        from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond

        LaunchpadWirebond(design, "atlas_start", options={"pos_x": "-2mm", "orientation": "0"})
        LaunchpadWirebond(design, "atlas_end", options={"pos_x": "2mm", "orientation": "180"})
        target = component_class(design, "atlas_target", options=_route_options(spec.class_name))
    else:
        target = component_class(design, "atlas_target")
    return design, target


def target_geometries(design: Any, component_id: int) -> tuple[Any, Any, dict[str, int]]:
    """Return target-owned conductor/subtractive unions in Metal millimetres."""
    conductors, etches = [], []
    counts = {"poly": 0, "path": 0, "junction": 0}
    for table_name in ("poly", "path"):
        table = design.qgeometry.tables[table_name]
        rows = table[table.component == component_id]
        counts[table_name] = int(len(rows))
        for row in rows.itertuples(index=False):
            if bool(row.helper):
                continue
            geometry = row.geometry
            if table_name == "path":
                geometry = geometry.buffer(float(row.width) / 2, cap_style=2, join_style=1)
            (etches if bool(row.subtract) else conductors).append(geometry)
    junctions = design.qgeometry.tables["junction"]
    counts["junction"] = int(len(junctions[junctions.component == component_id]))
    return unary_union(conductors), unary_union(etches), counts


def conductor_islands(conductor: Any) -> list[Any]:
    if conductor is None or conductor.is_empty:
        return []
    clean = conductor.buffer(0)
    islands = [part for part in getattr(clean, "geoms", [clean]) if part.area > 0]
    # Shape-first ordering is translation invariant.  Centroids only break ties;
    # symmetric unnamed islands remain explicitly marked as ambiguous below.
    return sorted(
        islands,
        key=lambda part: (
            round(float(part.area), 12),
            round(float(part.length), 12),
            hashlib.sha256(wkb_dumps(_centered(part))).hexdigest(),
            round(float(part.centroid.x), 12),
            round(float(part.centroid.y), 12),
        ),
    )


def _centered(geometry: Any) -> Any:
    left, bottom, right, top = geometry.bounds
    return translate(geometry, -0.5 * (left + right), -0.5 * (bottom + top))


def _pin_island(pin: Any, islands: list[Any], tolerance_mm: float = 1e-6) -> int | None:
    line = LineString(np.asarray(pin["points"], dtype=float))
    distances = [float(part.distance(line)) for part in islands]
    if not distances or min(distances) > tolerance_mm:
        return None
    return int(np.argmin(distances))


def assign_terminals(component: Any, islands: list[Any]) -> list[dict[str, Any]]:
    """Collapse all pins touching the same conductor and retain unmarked islands."""
    pin_groups: dict[int, list[str]] = {index: [] for index in range(len(islands))}
    unmatched = []
    for pin_name in sorted(component.pins):
        island = _pin_island(component.pins[pin_name], islands)
        if island is None:
            unmatched.append(pin_name)
        else:
            pin_groups[island].append(pin_name)
    terminals = []
    shape_keys = [
        (
            round(float(island.area), 12),
            round(float(island.length), 12),
            hashlib.sha256(wkb_dumps(_centered(island))).hexdigest(),
        )
        for island in islands
    ]
    for index, island in enumerate(islands):
        terminals.append(
            {
                "terminal_index": index,
                "terminal_id": f"terminal_{index}",
                "island_index": index,
                "pin_names": pin_groups[index],
                "assignment_method": "pin_to_island" if pin_groups[index] else "unmarked_conductor_island",
                "area_um2": float(island.area * MM_TO_UM**2),
                "perimeter_um": float(island.length * MM_TO_UM),
                "marker_layer": 2 + index,
                "marker_datatype": 0,
                "ordering_ambiguous": shape_keys.count(shape_keys[index]) > 1 and not pin_groups[index],
            }
        )
    if unmatched:
        # Pins with no metal contact are interfaces or semantic annotations, not
        # additional capacitance terminals.  Preserve them for auditability.
        for terminal in terminals:
            terminal["unmatched_component_pins"] = unmatched
    return terminals


def _pin_marker(pin: Any, length_mm: float) -> Polygon:
    points = np.asarray(pin["points"], dtype=float)
    normal = np.asarray(pin["normal"], dtype=float)
    normal = normal / np.linalg.norm(normal)
    extension = normal * length_mm
    return Polygon([points[0], points[1], points[1] + extension, points[0] + extension])


def _synthetic_marker(island: Any, conductor: Any, length_mm: float) -> Polygon:
    """Make a deterministic assignment bridge for an island lacking a pin."""
    center = np.asarray([conductor.centroid.x, conductor.centroid.y], dtype=float)
    local = np.asarray([island.centroid.x, island.centroid.y], dtype=float)
    direction = local - center
    if np.linalg.norm(direction) < 1e-12:
        direction = np.asarray([1.0, 0.0])
    direction /= np.linalg.norm(direction)
    far = Point(*(local + direction * max(conductor.bounds[2] - conductor.bounds[0], conductor.bounds[3] - conductor.bounds[1], 1.0)))
    start = nearest_points(island.boundary, far)[0]
    tangent = np.asarray([-direction[1], direction[0]])
    half_width = min(max(np.sqrt(float(island.area)) * 0.02, 0.0005), 0.0025)
    point = np.asarray([start.x, start.y])
    inward = direction * 1e-7
    return Polygon(
        [
            point - tangent * half_width - inward,
            point + tangent * half_width - inward,
            point + tangent * half_width + direction * length_mm,
            point - tangent * half_width + direction * length_mm,
        ]
    )


def terminal_markers(component: Any, islands: list[Any], terminals: list[dict[str, Any]], clearance_um: float):
    markers = []
    conductor = unary_union(islands)
    for terminal in terminals:
        pin_names = terminal["pin_names"]
        polygon = (
            _pin_marker(component.pins[pin_names[0]], clearance_um / MM_TO_UM)
            if pin_names
            else _synthetic_marker(islands[terminal["island_index"]], conductor, clearance_um / MM_TO_UM)
        )
        markers.append((terminal["marker_layer"], terminal["marker_datatype"], polygon))
    return markers


def _gap_values(options: Any):
    try:
        items = options.items()
    except AttributeError:
        return
    for key, value in items:
        if hasattr(value, "items"):
            yield from _gap_values(value)
        elif "gap" in str(key).lower() and isinstance(value, str | int | float):
            yield value


def clearance(component: Any) -> tuple[float, str]:
    values = []
    for raw in _gap_values(component.options):
        try:
            parsed = float(component.design.parse_value(raw)) * MM_TO_UM
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            values.append(parsed)
    return (min(values), "minimum_positive_gap_option") if values else (DEFAULT_CLEARANCE_UM, "atlas_fixture_default")


def _connected_hole(geometry: Any) -> Any:
    hole = geometry.buffer(0)
    while hole.geom_type == "MultiPolygon":
        parts = list(hole.geoms)
        candidates = []
        for first in range(len(parts)):
            for second in range(first + 1, len(parts)):
                candidates.append((float(parts[first].distance(parts[second])), first, second))
        _, first, second = min(candidates)
        start, end = nearest_points(parts[first], parts[second])
        bridge = LineString([start, end]).buffer(1e-6, cap_style=3, join_style=2)
        hole = unary_union([hole, bridge]).buffer(0)
    if hole.geom_type != "Polygon" or not hole.is_valid or hole.is_empty:
        raise ValueError("Could not construct one valid connected ground hole.")
    # Interior rings would become isolated ground islands inside the one
    # simulation opening.  The atlas contract deliberately has one simply
    # connected hole, so fill them here.
    return Polygon(hole.exterior)


def ground_geometry(conductor: Any, etch: Any, markers: list[tuple[int, int, Any]], clearance_um: float):
    clearance_mm = clearance_um / MM_TO_UM
    pieces = [conductor.buffer(clearance_mm, join_style=2), *(polygon for _, _, polygon in markers)]
    if etch is not None and not etch.is_empty:
        pieces.append(etch)
    hole = _connected_hole(unary_union(pieces))
    left, bottom, right, top = conductor.bounds
    cx, cy = 0.5 * (left + right), 0.5 * (bottom + top)
    half = GROUND_PADDING_UM / MM_TO_UM + 0.5 * max(right - left, top - bottom)
    hl, hb, hr, ht = hole.bounds
    padding_mm = GROUND_PADDING_UM / MM_TO_UM
    half = max(
        half,
        abs(hl - cx) + padding_mm,
        abs(hr - cx) + padding_mm,
        abs(hb - cy) + padding_mm,
        abs(ht - cy) + padding_mm,
    )
    domain = box(cx - half, cy - half, cx + half, cy + half)
    return domain.difference(hole), hole


def _insert(cell: Any, layer_index: int, geometry: Any, kdb: Any):
    for polygon in getattr(geometry, "geoms", [geometry]):
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        shell = [kdb.DPoint(float(x) * MM_TO_UM, float(y) * MM_TO_UM) for x, y in polygon.exterior.coords[:-1]]
        target = kdb.DPolygon(shell)
        for interior in polygon.interiors:
            target.insert_hole([kdb.DPoint(float(x) * MM_TO_UM, float(y) * MM_TO_UM) for x, y in interior.coords[:-1]])
        cell.shapes(layer_index).insert(target)


def export_gds(path: Path, conductor: Any, ground: Any, markers: list[tuple[int, int, Any]]):
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.dbu = 0.001
    cell = layout.create_cell("TOP")
    _insert(cell, layout.layer(1, 10), conductor, kdb)
    _insert(cell, layout.layer(1, 0), ground, kdb)
    for layer, datatype, polygon in markers:
        _insert(cell, layout.layer(layer, datatype), polygon, kdb)
    options = kdb.SaveLayoutOptions()
    options.gds2_write_timestamps = False
    # Keep XY records below the legacy signed-16-bit reader boundary.  KLayout
    # fractures more complex polygons without changing their merged region.
    options.gds2_max_vertex_count = 4000
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.gds")
    layout.write(str(temporary), options)
    temporary.replace(path)


def write_layer_contract(path: Path, terminals: list[dict[str, Any]]) -> Path:
    """Write semantic terminal-to-conductor bindings beside one atlas GDS."""
    sidecar = {
        "schema_version": "squadds-planar-layer-contract-v1",
        "gds_file": path.name,
        "units": "um",
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
        "layers": [
            {"layer": 1, "datatype": 0, "role": "ground", "plane_id": "device"},
            {"layer": 1, "datatype": 10, "role": "conductor", "plane_id": "device"},
            *[
                {
                    "layer": terminal["marker_layer"],
                    "datatype": terminal["marker_datatype"],
                    "role": "port",
                    "plane_id": "device",
                    "terminal_index": terminal["terminal_index"],
                    "terminal_id": terminal["terminal_id"],
                    "net_id": terminal["terminal_id"],
                    "conductor_area_um2": terminal["area_um2"],
                    "conductor_perimeter_um": terminal["perimeter_um"],
                }
                for terminal in terminals
            ],
        ],
        "future_stack_hook": {
            "implemented_geometry": "planar",
            "material_stack_file": None,
        },
    }
    sidecar_path = path.with_suffix(".layers.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return sidecar_path


def validate_gds(
    path: Path,
    conductor: Any,
    ground: Any,
    markers: list[tuple[int, int, Any]],
) -> dict[str, Any]:
    """Read an artifact back and verify the complete atlas layer contract."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.top_cell()
    if top is None:
        raise ValueError("Exported GDS has no top cell.")
    dbu = float(layout.dbu)
    regions: dict[tuple[int, int], Any] = {}
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        region = kdb.Region(top.begin_shapes_rec(layer_index))
        region.merge()
        regions[(int(info.layer), int(info.datatype))] = region
    expected_layers = {(1, 0), (1, 10), *((layer, datatype) for layer, datatype, _ in markers)}
    conductor_area = float(regions.get((1, 10), kdb.Region()).area() * dbu**2)
    ground_area = float(regions.get((1, 0), kdb.Region()).area() * dbu**2)
    expected_conductor_area = float(conductor.area * MM_TO_UM**2)
    expected_ground_area = float(ground.area * MM_TO_UM**2)
    ground_polygons = list(regions.get((1, 0), kdb.Region()).each())
    checks = {
        "top_cell_is_TOP": top.name == "TOP",
        "dbu_is_1nm": abs(dbu - 0.001) <= 1e-12,
        "exact_layer_set": set(regions) == expected_layers,
        "all_terminal_marker_layers_present": all((layer, datatype) in regions for layer, datatype, _ in markers),
        "conductor_area_roundtrip": abs(conductor_area - expected_conductor_area)
        <= max(0.01, expected_conductor_area * 1e-5),
        "ground_area_roundtrip": abs(ground_area - expected_ground_area)
        <= max(0.1, expected_ground_area * 1e-5),
        "one_ground_polygon": len(ground_polygons) == 1,
        "one_ground_hole": len(ground_polygons) == 1 and ground_polygons[0].holes() == 1,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "dbu_um": dbu,
        "layers": [list(key) for key in sorted(regions)],
        "conductor_area_um2": conductor_area,
        "ground_area_um2": ground_area,
    }


def geometry_fingerprint(conductor: Any, etch: Any) -> str:
    """Hash physical target geometry, deliberately excluding labels and fixtures."""
    complete = unary_union([conductor, etch] if etch is not None and not etch.is_empty else [conductor])
    left, bottom, right, top = complete.bounds
    dx, dy = -0.5 * (left + right), -0.5 * (bottom + top)
    payload = wkb_dumps(translate(conductor, dx, dy))
    if etch is not None and not etch.is_empty:
        payload += wkb_dumps(translate(etch, dx, dy))
    return f"geometry:sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_one(spec: ComponentSpec, output_dir: Path) -> dict[str, Any]:
    base = {
        "component": asdict(spec),
        "inventory_version": INVENTORY_VERSION,
        "status": "pending",
        "generation": {
            "recipe_kind": spec.recipe,
            "options": _route_options(spec.class_name) if spec.recipe == "contextual_route" else {},
            "fixture_components": (
                [
                    {"class_name": "LaunchpadWirebond", "name": "atlas_start", "pos_x": "-2mm"},
                    {"class_name": "LaunchpadWirebond", "name": "atlas_end", "pos_x": "2mm"},
                ]
                if spec.recipe == "contextual_route"
                else []
            ),
            "exports_target_component_only": True,
        },
    }
    try:
        component_class = _component_class(spec)
        base["module_source_sha256"] = _source_sha256(component_class)
        design, component = instantiate(spec)
        conductor, etch, counts = target_geometries(design, component.id)
        base["qgeometry_counts"] = counts
        if conductor is None or conductor.is_empty:
            base.update(
                status="semantic_only_requires_context",
                pins=sorted(component.pins),
                reason=(
                    "The target emits no standalone conductor poly/path qgeometry; "
                    "its meaning requires connected surrounding geometry."
                ),
            )
            return base
        islands = conductor_islands(conductor)
        terminals = assign_terminals(component, islands)
        clearance_um, clearance_source = clearance(component)
        markers = terminal_markers(component, islands, terminals, clearance_um)
        ground, _ = ground_geometry(conductor, etch, markers, clearance_um)
        relative_path = Path("gds") / f"{spec.category}__{spec.class_name}.gds"
        artifact = output_dir / relative_path
        export_gds(artifact, conductor, ground, markers)
        layer_contract_path = write_layer_contract(artifact, terminals)
        validation = validate_gds(artifact, conductor, ground, markers)
        if not validation["valid"]:
            raise ValueError(f"GDS readback validation failed: {validation['checks']}")
        base.update(
            status="generated_valid",
            gds_path=relative_path.as_posix(),
            layer_contract_path=layer_contract_path.relative_to(output_dir).as_posix(),
            artifact_sha256=_sha256_file(artifact),
            artifact_size_bytes=artifact.stat().st_size,
            geometry_id=geometry_fingerprint(conductor, etch),
            conductor_island_count=len(islands),
            terminal_count=len(terminals),
            terminals=terminals,
            component_pins=sorted(component.pins),
            clearance_um=clearance_um,
            clearance_source=clearance_source,
            validation=validation,
            layers=[
                {"layer": 1, "datatype": 0, "role": "ground_simulation_domain"},
                {"layer": 1, "datatype": 10, "role": "signal_conductors"},
                *[
                    {"layer": item["marker_layer"], "datatype": 0, "role": item["terminal_id"]}
                    for item in terminals
                ],
            ],
        )
    except Exception as error:  # noqa: BLE001 - one catalog entry must not stop the audit.
        base.update(status="generation_failure", error_type=type(error).__name__, error_message=str(error))
    return base


def build_atlas(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [build_one(spec, output_dir) for spec in COMPONENT_SPECS]
    groups: dict[str, list[str]] = {}
    for record in records:
        if record["status"] == "generated_valid":
            groups.setdefault(record["geometry_id"], []).append(record["component"]["class_name"])
    equivalence_classes = [
        {"geometry_id": key, "members": sorted(members), "size": len(members)}
        for key, members in sorted(groups.items())
    ]
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    first_class = _component_class(COMPONENT_SPECS[0])
    payload = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "inventory_version": INVENTORY_VERSION,
        "inventory_count": len(COMPONENT_SPECS),
        "qiskit_metal": _git_metadata(first_class),
        "counts": {
            "by_status": by_status,
            "geometry_equivalence_classes": len(equivalence_classes),
            "non_singleton_equivalence_classes": sum(item["size"] > 1 for item in equivalence_classes),
        },
        "geometry_equivalence_classes": equivalence_classes,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("tutorials/runtime/tutorial24/qmetal-atlas"),
    )
    arguments = parser.parse_args()
    result = build_atlas(arguments.output_dir)
    print(json.dumps({"output": str(arguments.output_dir), **result["counts"]}, indent=2))
    if result["counts"]["by_status"].get("generation_failure"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
