"""Extract electrical-net geometry from GDS using explicit sidecar metadata.

GDSII stores layers and polygons, not the semantic order of a capacitance
matrix.  TopoCap therefore keeps that order in a JSON sidecar.  The contract is
plain data rather than a second graph-schema class so it can be validated,
hashed, uploaded, and adapted to :mod:`squadds.ml.topocap.schema` independently.

The sidecar shape is::

    {
      "schema_version": "topocap-net-sidecar-1.0.0",
      "gds": {"sha256": "...", "top_cell": "main", "coordinate_unit": "um"},
      "matrix": {
        "convention": "signed_maxwell",
        "units": "fF",
        "node_order": ["net_000", "net_001", "net_002"]
      },
      "nets": [
        {
          "net_id": "net_000",
          "is_reference": true,
          "geometry_selectors": [
            {"layer": 1, "datatype": 0, "selection": "all"}
          ],
          "port_selectors": []
        }
      ],
      "auxiliary_geometry": [
        {
          "role": "dielectric_etch",
          "selectors": [
            {"layer": 1, "datatype": 11, "selection": "all"}
          ]
        }
      ]
    }

Selectors use exactly one of ``all``, ``polygon_indices``, or
``component_indices``.  Polygon indices refer to the exact, immutable order in
the hashed GDS artifact.  Component indices refer to deterministically sorted
connected components after same-layer union.  Neither form assigns electrical
meaning by itself; ``matrix.node_order`` and the per-net selectors do that.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

NET_SIDECAR_SCHEMA_VERSION = "topocap-net-sidecar-1.0.0"
MATRIX_CONVENTION = "signed_maxwell"
CANONICAL_CAPACITANCE_UNIT = "fF"


class NetSidecarError(ValueError):
    """Raised when a net sidecar is incomplete or internally inconsistent."""


class NetMappingError(NetSidecarError):
    """Raised when geometry cannot be aligned to an explicit matrix contract."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def content_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using its canonical serialization."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one artifact without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_geometry_dependencies():
    try:
        import gdstk
        import shapely
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("TopoCap GDS extraction requires gdstk and shapely>=2.0.") from exc
    return gdstk, shapely, Polygon, MultiPolygon, GeometryCollection, unary_union


def _polygonal_parts(geometry: Any) -> list[Any]:
    """Return polygonal members of a potentially repaired Shapely geometry."""
    if geometry.is_empty:
        return []
    geometry_type = geometry.geom_type
    if geometry_type == "Polygon":
        return [geometry]
    if geometry_type in {"MultiPolygon", "GeometryCollection"}:
        parts: list[Any] = []
        for member in geometry.geoms:
            parts.extend(_polygonal_parts(member))
        return parts
    return []


def _repair_polygon(points_um: Any) -> Any:
    """Convert one GDS polygon to valid polygonal Shapely geometry."""
    _, shapely, Polygon, _, _, unary_union = _require_geometry_dependencies()
    polygon = Polygon(points_um)
    if polygon.is_empty:
        raise NetMappingError("GDS contains an empty polygon.")
    repaired = polygon if polygon.is_valid else shapely.make_valid(polygon)
    parts = _polygonal_parts(repaired)
    if not parts:
        raise NetMappingError("A GDS polygon could not be repaired into polygonal geometry.")
    return unary_union(parts)


def _geometry_key(geometry: Any) -> tuple[Any, ...]:
    """Build a stable ordering key for connected components."""
    _, shapely, _, _, _, _ = _require_geometry_dependencies()
    minimum_x, minimum_y, maximum_x, maximum_y = geometry.bounds
    normalized = shapely.normalize(geometry)
    return (
        round(float(minimum_x), 12),
        round(float(minimum_y), 12),
        round(float(maximum_x), 12),
        round(float(maximum_y), 12),
        round(float(geometry.area), 12),
        hashlib.sha256(normalized.wkb).hexdigest(),
    )


def read_gds_inventory(path: str | Path, *, top_cell: str | None = None) -> dict[str, Any]:
    """Read one GDS into a layer inventory expressed in physical micrometres.

    Raw polygons retain their immutable GDS order.  Same-layer connected
    components are also exposed in a deterministic geometry-derived order.
    Only one GDS is resident while this function runs.
    """
    gdstk, _, _, _, _, unary_union = _require_geometry_dependencies()
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    library = gdstk.read_gds(str(artifact_path))
    scale_um = float(library.unit) * 1.0e6
    if not math.isfinite(scale_um) or scale_um <= 0.0:
        raise NetMappingError(f"Invalid GDS user unit in {artifact_path}: {library.unit!r}")

    top_cells = {cell.name: cell for cell in library.top_level()}
    if top_cell is None:
        if len(top_cells) != 1:
            names = sorted(top_cells)
            raise NetMappingError(f"GDS must have one top cell or an explicit top_cell; found {names}.")
        selected_cell = next(iter(top_cells.values()))
    else:
        try:
            selected_cell = top_cells[top_cell]
        except KeyError as exc:
            raise NetMappingError(f"Top cell {top_cell!r} is not present in {artifact_path}.") from exc

    raw_by_layer: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for gds_polygon in selected_cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
        points_um = gds_polygon.points.astype(float, copy=False) * scale_um
        geometry = _repair_polygon(points_um)
        raw_by_layer[(int(gds_polygon.layer), int(gds_polygon.datatype))].append(geometry)

    layers: dict[tuple[int, int], dict[str, Any]] = {}
    for layer_spec, polygons in sorted(raw_by_layer.items()):
        merged = unary_union(polygons)
        components = sorted(_polygonal_parts(merged), key=_geometry_key)
        layers[layer_spec] = {
            "polygons": tuple(polygons),
            "components": tuple(components),
        }
    if not layers:
        raise NetMappingError(f"No polygonal geometry found in {artifact_path}.")
    return {
        "path": artifact_path,
        "sha256": sha256_file(artifact_path),
        "top_cell": selected_cell.name,
        "coordinate_scale_um": scale_um,
        "layers": layers,
    }


def inventory_summary(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable summary suitable for audit logs."""
    layers = []
    for (layer, datatype), values in sorted(inventory["layers"].items()):
        polygons = values["polygons"]
        components = values["components"]
        layers.append(
            {
                "layer": int(layer),
                "datatype": int(datatype),
                "polygon_count": len(polygons),
                "component_count": len(components),
                "area_um2": float(sum(geometry.area for geometry in polygons)),
            }
        )
    return {
        "sha256": str(inventory["sha256"]),
        "top_cell": str(inventory["top_cell"]),
        "coordinate_scale_um": float(inventory["coordinate_scale_um"]),
        "layers": layers,
    }


def _selector(layer: int, datatype: int, selection: str, indices: Sequence[int] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "layer": int(layer),
        "datatype": int(datatype),
        "selection": selection,
    }
    if indices is not None:
        result["indices"] = [int(index) for index in indices]
    return result


def _validate_selector(selector: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    required = {"layer", "datatype", "selection"}
    missing = required.difference(selector)
    if missing:
        raise NetSidecarError(f"{context} is missing selector fields: {sorted(missing)}")
    try:
        layer = int(selector["layer"])
        datatype = int(selector["datatype"])
    except (TypeError, ValueError) as exc:
        raise NetSidecarError(f"{context} layer/datatype must be integers.") from exc
    selection = str(selector["selection"])
    if selection not in {"all", "polygon_indices", "component_indices"}:
        raise NetSidecarError(f"{context} has unsupported selection {selection!r}.")
    indices = selector.get("indices")
    if selection == "all":
        if indices not in (None, []):
            raise NetSidecarError(f"{context} selection='all' cannot include indices.")
        return _selector(layer, datatype, selection)
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)) or not indices:
        raise NetSidecarError(f"{context} requires a non-empty indices list.")
    normalized_indices = []
    for index in indices:
        if isinstance(index, bool):
            raise NetSidecarError(f"{context} indices must be non-negative integers.")
        try:
            normalized = int(index)
        except (TypeError, ValueError) as exc:
            raise NetSidecarError(f"{context} indices must be non-negative integers.") from exc
        if normalized < 0 or normalized != index:
            raise NetSidecarError(f"{context} indices must be non-negative integers.")
        normalized_indices.append(normalized)
    if len(set(normalized_indices)) != len(normalized_indices):
        raise NetSidecarError(f"{context} contains duplicate indices.")
    return _selector(layer, datatype, selection, normalized_indices)


def validate_net_sidecar(sidecar: Mapping[str, Any], *, inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and return a normalized JSON-compatible sidecar."""
    if sidecar.get("schema_version") != NET_SIDECAR_SCHEMA_VERSION:
        raise NetSidecarError(f"Unsupported net sidecar schema: {sidecar.get('schema_version')!r}")
    gds = sidecar.get("gds")
    matrix = sidecar.get("matrix")
    nets = sidecar.get("nets")
    if not isinstance(gds, Mapping) or not isinstance(matrix, Mapping) or not isinstance(nets, Sequence):
        raise NetSidecarError("Sidecar requires gds, matrix, and nets sections.")
    if matrix.get("convention") != MATRIX_CONVENTION:
        raise NetSidecarError(f"matrix.convention must be {MATRIX_CONVENTION!r}.")
    if matrix.get("units") != CANONICAL_CAPACITANCE_UNIT:
        raise NetSidecarError("Sidecar matrices must be normalized explicitly to fF.")
    node_order = matrix.get("node_order")
    if not isinstance(node_order, Sequence) or isinstance(node_order, (str, bytes)) or len(node_order) < 2:
        raise NetSidecarError("matrix.node_order must list at least two net IDs.")
    node_order = [str(net_id) for net_id in node_order]
    if any(not net_id for net_id in node_order) or len(set(node_order)) != len(node_order):
        raise NetSidecarError("matrix.node_order must contain unique non-empty net IDs.")

    normalized_nets = []
    for net_index, net in enumerate(nets):
        if not isinstance(net, Mapping):
            raise NetSidecarError(f"nets[{net_index}] must be a mapping.")
        net_id = str(net.get("net_id", ""))
        if not net_id:
            raise NetSidecarError(f"nets[{net_index}].net_id must be non-empty.")
        geometry_selectors = net.get("geometry_selectors")
        port_selectors = net.get("port_selectors", [])
        if not isinstance(geometry_selectors, Sequence) or not geometry_selectors:
            raise NetSidecarError(f"Net {net_id!r} requires geometry_selectors.")
        if not isinstance(port_selectors, Sequence):
            raise NetSidecarError(f"Net {net_id!r} port_selectors must be a list.")
        normalized_nets.append(
            {
                "net_id": net_id,
                "is_reference": bool(net.get("is_reference", False)),
                "geometry_selectors": [
                    _validate_selector(value, context=f"net {net_id!r} geometry selector")
                    for value in geometry_selectors
                ],
                "port_selectors": [
                    _validate_selector(value, context=f"net {net_id!r} port selector") for value in port_selectors
                ],
            }
        )
    net_ids = [net["net_id"] for net in normalized_nets]
    if len(set(net_ids)) != len(net_ids):
        raise NetSidecarError("nets contains duplicate net IDs.")
    if net_ids != node_order:
        raise NetSidecarError("nets must appear in exactly matrix.node_order; implicit reordering is forbidden.")

    auxiliary = sidecar.get("auxiliary_geometry", [])
    if not isinstance(auxiliary, Sequence):
        raise NetSidecarError("auxiliary_geometry must be a list.")
    normalized_auxiliary = []
    for index, entry in enumerate(auxiliary):
        if not isinstance(entry, Mapping) or not str(entry.get("role", "")):
            raise NetSidecarError(f"auxiliary_geometry[{index}] requires a non-empty role.")
        selectors = entry.get("selectors")
        if not isinstance(selectors, Sequence) or not selectors:
            raise NetSidecarError(f"auxiliary_geometry[{index}] requires selectors.")
        normalized_auxiliary.append(
            {
                "role": str(entry["role"]),
                "selectors": [
                    _validate_selector(value, context=f"auxiliary_geometry[{index}] selector") for value in selectors
                ],
            }
        )

    digest = str(gds.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise NetSidecarError("gds.sha256 must be a hexadecimal SHA-256 digest.")
    normalized: dict[str, Any] = {
        "schema_version": NET_SIDECAR_SCHEMA_VERSION,
        "gds": {
            "sha256": digest.lower(),
            "top_cell": str(gds.get("top_cell", "")),
            "coordinate_unit": str(gds.get("coordinate_unit", "")),
        },
        "matrix": {
            "convention": MATRIX_CONVENTION,
            "units": CANONICAL_CAPACITANCE_UNIT,
            "node_order": node_order,
        },
        "nets": normalized_nets,
        "auxiliary_geometry": normalized_auxiliary,
        "provenance": dict(sidecar.get("provenance", {})),
    }
    if normalized["gds"]["coordinate_unit"] != "um":
        raise NetSidecarError("gds.coordinate_unit must be 'um'.")
    if not normalized["gds"]["top_cell"]:
        raise NetSidecarError("gds.top_cell must be explicit.")

    if inventory is not None:
        if digest.lower() != str(inventory["sha256"]).lower():
            raise NetSidecarError("Sidecar GDS hash does not match the supplied artifact.")
        if normalized["gds"]["top_cell"] != inventory["top_cell"]:
            raise NetSidecarError("Sidecar top cell does not match the supplied artifact.")
        for net in normalized_nets:
            for selector in [*net["geometry_selectors"], *net["port_selectors"]]:
                _validate_selector_against_inventory(selector, inventory)
        for entry in normalized_auxiliary:
            for selector in entry["selectors"]:
                _validate_selector_against_inventory(selector, inventory)

    # Prove that the entire result remains JSON serializable and finite.
    canonical_json(normalized)
    normalized["sidecar_sha256"] = content_sha256(normalized)
    return normalized


def _validate_selector_against_inventory(selector: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    layer_spec = (int(selector["layer"]), int(selector["datatype"]))
    try:
        layer = inventory["layers"][layer_spec]
    except KeyError as exc:
        raise NetSidecarError(f"Selector references absent layer/datatype {layer_spec}.") from exc
    selection = selector["selection"]
    if selection == "all":
        return
    collection_name = "polygons" if selection == "polygon_indices" else "components"
    size = len(layer[collection_name])
    bad = [index for index in selector["indices"] if index >= size]
    if bad:
        raise NetSidecarError(f"Selector {selection} indices {bad} exceed {layer_spec} size {size}.")


def select_geometry(inventory: Mapping[str, Any], selectors: Sequence[Mapping[str, Any]]) -> Any:
    """Resolve validated selectors to a Shapely union."""
    _, _, _, _, _, unary_union = _require_geometry_dependencies()
    selected = []
    for raw_selector in selectors:
        selector = _validate_selector(raw_selector, context="geometry selector")
        _validate_selector_against_inventory(selector, inventory)
        layer = inventory["layers"][(selector["layer"], selector["datatype"])]
        if selector["selection"] == "all":
            selected.extend(layer["polygons"])
        else:
            collection = layer["polygons"] if selector["selection"] == "polygon_indices" else layer["components"]
            selected.extend(collection[index] for index in selector["indices"])
    if not selected:
        raise NetMappingError("Selectors resolved to no geometry.")
    return unary_union(selected)


def extract_sidecar_geometry(
    path: str | Path,
    sidecar: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve all sidecar nets, ports, and auxiliary roles for one GDS."""
    inventory = (
        read_gds_inventory(path, top_cell=sidecar.get("gds", {}).get("top_cell")) if inventory is None else inventory
    )
    normalized = validate_net_sidecar(sidecar, inventory=inventory)
    nets = []
    for net in normalized["nets"]:
        ports = [select_geometry(inventory, [selector]) for selector in net["port_selectors"]]
        nets.append(
            {
                "net_id": net["net_id"],
                "is_reference": net["is_reference"],
                "geometry": select_geometry(inventory, net["geometry_selectors"]),
                "port_geometries": ports,
            }
        )
    auxiliary = [
        {
            "role": entry["role"],
            "geometry": select_geometry(inventory, entry["selectors"]),
        }
        for entry in normalized["auxiliary_geometry"]
    ]
    return {
        "sidecar": normalized,
        "inventory": inventory,
        "nets": nets,
        "auxiliary_geometry": auxiliary,
    }


def _base_sidecar(
    inventory: Mapping[str, Any],
    nets: Sequence[Mapping[str, Any]],
    *,
    auxiliary_geometry: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    node_order = [str(net["net_id"]) for net in nets]
    candidate = {
        "schema_version": NET_SIDECAR_SCHEMA_VERSION,
        "gds": {
            "sha256": inventory["sha256"],
            "top_cell": inventory["top_cell"],
            "coordinate_unit": "um",
        },
        "matrix": {
            "convention": MATRIX_CONVENTION,
            "units": CANONICAL_CAPACITANCE_UNIT,
            "node_order": node_order,
        },
        "nets": list(nets),
        "auxiliary_geometry": list(auxiliary_geometry),
        "provenance": dict(provenance or {}),
    }
    return validate_net_sidecar(candidate, inventory=inventory)


def _nearest_unique_components(marker_geometries: Sequence[Any], signal_components: Sequence[Any]) -> tuple[int, ...]:
    if len(marker_geometries) != len(signal_components):
        raise NetMappingError("Port-marker count must equal signal-component count for marker-based alignment.")
    assignments = []
    for marker in marker_geometries:
        distances = [float(marker.distance(component)) for component in signal_components]
        assignments.append(min(range(len(distances)), key=lambda index: (distances[index], index)))
    if len(set(assignments)) != len(assignments):
        raise NetMappingError("Port markers do not map uniquely to signal components.")
    return tuple(assignments)


def build_generalized_ncap_sidecar(
    path: str | Path,
    *,
    inventory: Mapping[str, Any] | None = None,
    signal_component_order: Sequence[int] | None = None,
    source_id: str | None = None,
    design_id: str | None = None,
) -> dict[str, Any]:
    """Build the observed GeneralizedNCap sidecar without encoding net names.

    Matrix order is ``reference, marker-layer-2 signal, marker-layer-3 signal``.
    When marker layers are absent, callers must provide the two signal
    component indices explicitly; geometry position is never treated as a
    semantic matrix label.
    """
    inventory = read_gds_inventory(path) if inventory is None else inventory
    signal_layer = inventory["layers"].get((1, 10))
    ground_layer = inventory["layers"].get((1, 0))
    if signal_layer is None or len(signal_layer["components"]) != 2:
        raise NetMappingError("GeneralizedNCap requires exactly two connected signal components on (1, 10).")
    if ground_layer is None or not ground_layer["polygons"]:
        raise NetMappingError("GeneralizedNCap requires explicit ground/domain geometry on (1, 0).")

    port_specs = ((2, 0), (3, 0))
    marker_geometries = []
    markers_present = all(spec in inventory["layers"] for spec in port_specs)
    if markers_present:
        for spec in port_specs:
            marker_geometries.append(select_geometry(inventory, [_selector(*spec, "all")]))
        inferred_order = _nearest_unique_components(marker_geometries, signal_layer["components"])
        if signal_component_order is not None and tuple(signal_component_order) != inferred_order:
            raise NetMappingError("Explicit signal order disagrees with the port-marker alignment.")
        signal_component_order = inferred_order
        mapping_basis = "explicit_port_marker_layer_order"
    elif signal_component_order is None:
        raise NetMappingError(
            "GeneralizedNCap GDS has no complete port-marker pair; supply signal_component_order explicitly."
        )
    else:
        mapping_basis = "caller_supplied_component_order"

    order = tuple(int(index) for index in signal_component_order)
    if sorted(order) != [0, 1]:
        raise NetMappingError("signal_component_order must be a permutation of (0, 1).")
    nets = [
        {
            "net_id": "net_000",
            "is_reference": True,
            "geometry_selectors": [_selector(1, 0, "all")],
            "port_selectors": [],
        }
    ]
    for matrix_offset, component_index in enumerate(order, start=1):
        port_selectors = []
        if markers_present:
            port_selectors = [_selector(*port_specs[matrix_offset - 1], "all")]
        nets.append(
            {
                "net_id": f"net_{matrix_offset:03d}",
                "is_reference": False,
                "geometry_selectors": [_selector(1, 10, "component_indices", [component_index])],
                "port_selectors": port_selectors,
            }
        )
    return _base_sidecar(
        inventory,
        nets,
        provenance={
            "mapping_basis": mapping_basis,
            "source_id": source_id,
            "design_id": design_id,
        },
    )


def build_capn_interdigital_tee_sidecar(
    path: str | Path,
    *,
    signal_polygon_order: Sequence[int],
    inventory: Mapping[str, Any] | None = None,
    source_id: str | None = None,
    design_id: str | None = None,
) -> dict[str, Any]:
    """Build a legacy CapN sidecar from an explicit generator-order contract.

    The GDS has no port markers, so ``signal_polygon_order`` is mandatory.  For
    the immutable SQuADDS generator artifacts it is ``(0, 1)``, corresponding
    to the Qiskit-Metal ``cap_body_0``/``cap_body_1`` solver ordering.  That
    assumption is recorded and protected by the exact GDS hash.
    """
    inventory = read_gds_inventory(path) if inventory is None else inventory
    signal_layer = inventory["layers"].get((1, 10))
    ground_layer = inventory["layers"].get((1, 0))
    if signal_layer is None or len(signal_layer["polygons"]) != 2:
        raise NetMappingError("CapNInterdigitalTee requires exactly two signal polygons on (1, 10).")
    if ground_layer is None or not ground_layer["polygons"]:
        raise NetMappingError("CapNInterdigitalTee requires explicit ground/domain geometry on (1, 0).")
    order = tuple(int(index) for index in signal_polygon_order)
    if sorted(order) != [0, 1]:
        raise NetMappingError("signal_polygon_order must be a permutation of (0, 1).")

    nets = [
        {
            "net_id": "net_000",
            "is_reference": True,
            "geometry_selectors": [_selector(1, 0, "all")],
            "port_selectors": [],
        }
    ]
    for matrix_offset, polygon_index in enumerate(order, start=1):
        nets.append(
            {
                "net_id": f"net_{matrix_offset:03d}",
                "is_reference": False,
                "geometry_selectors": [_selector(1, 10, "polygon_indices", [polygon_index])],
                "port_selectors": [],
            }
        )
    auxiliary = []
    if (1, 11) in inventory["layers"]:
        auxiliary.append(
            {
                "role": "dielectric_etch",
                "selectors": [_selector(1, 11, "all")],
            }
        )
    return _base_sidecar(
        inventory,
        nets,
        auxiliary_geometry=auxiliary,
        provenance={
            "mapping_basis": "immutable_qiskit_metal_generator_polygon_order",
            "source_id": source_id,
            "design_id": design_id,
            "signal_polygon_order": list(order),
        },
    )


__all__ = [
    "CANONICAL_CAPACITANCE_UNIT",
    "MATRIX_CONVENTION",
    "NET_SIDECAR_SCHEMA_VERSION",
    "NetMappingError",
    "NetSidecarError",
    "build_capn_interdigital_tee_sidecar",
    "build_generalized_ncap_sidecar",
    "canonical_json",
    "content_sha256",
    "extract_sidecar_geometry",
    "inventory_summary",
    "read_gds_inventory",
    "select_geometry",
    "sha256_file",
    "validate_net_sidecar",
]
