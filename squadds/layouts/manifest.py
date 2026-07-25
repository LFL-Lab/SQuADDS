"""Build portable metadata records for immutable GDS layout artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_LAYOUT_REPOSITORY = "SQuADDS/SQuADDS_Layouts"
MANIFEST_SCHEMA_VERSION = "1.0.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_design_id(component_name: str, design_options: dict[str, Any]) -> str:
    """Return a stable ID for a parametric component design."""
    payload = {"component_name": component_name, "design_options": design_options}
    return f"design:sha256:{_sha256(_canonical_json(payload).encode())}"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of an exact artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_klayout():
    try:
        import klayout.db as kdb
    except ImportError as exc:
        raise ImportError("GDS inspection requires the optional dependency group: uv sync --extra gds") from exc
    return kdb


def _bbox_um(box, dbu_um: float) -> dict[str, float] | None:
    if box.empty():
        return None
    return {
        "left": box.left * dbu_um,
        "bottom": box.bottom * dbu_um,
        "right": box.right * dbu_um,
        "top": box.top * dbu_um,
    }


def _canonical_polygon(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Normalize polygon orientation and starting vertex for geometry hashing."""
    if not points:
        return ()
    rotations = []
    for sequence in (points, list(reversed(points))):
        rotations.extend(tuple(sequence[index:] + sequence[:index]) for index in range(len(sequence)))
    return min(rotations)


def _layout_id(layout, top_cell, dbu_um: float) -> str:
    layers = []
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        region = _require_klayout().Region(top_cell.begin_shapes_rec(layer_index))
        polygons = []
        for polygon in region.each():
            points = [(point.x, point.y) for point in polygon.each_point_hull()]
            polygons.append(_canonical_polygon(points))
        layers.append({"layer": info.layer, "datatype": info.datatype, "polygons": sorted(polygons)})
    payload = {"dbu_um": dbu_um, "layers": sorted(layers, key=lambda item: (item["layer"], item["datatype"]))}
    return f"layout:sha256:{_sha256(_canonical_json(payload).encode())}"


def parse_gds_summary(path: str | Path) -> dict[str, Any]:
    """Extract deterministic geometry metadata from a GDS file in micrometers."""
    kdb = _require_klayout()
    layout = kdb.Layout()
    layout.read(str(path))
    top_cell = layout.top_cell()
    if top_cell is None:
        raise ValueError(f"GDS file has no top cell: {path}")

    dbu_um = float(layout.dbu)
    layers = []
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        region = kdb.Region(top_cell.begin_shapes_rec(layer_index))
        layers.append(
            {
                "layer": int(info.layer),
                "datatype": int(info.datatype),
                "polygon_count": int(region.count()),
                "area_um2": float(region.area() * dbu_um**2),
                "bbox_um": _bbox_um(region.bbox(), dbu_um),
            }
        )

    return {
        "layout_id": _layout_id(layout, top_cell, dbu_um),
        "top_cell": top_cell.name,
        "cell_count": int(layout.cells()),
        "dbu_um": dbu_um,
        "bbox_um": _bbox_um(top_cell.bbox(), dbu_um),
        "layers": sorted(layers, key=lambda item: (item["layer"], item["datatype"])),
        "polygon_count": sum(item["polygon_count"] for item in layers),
    }


def parse_gds_polygons(
    path: str | Path,
    *,
    layer: int | None = None,
    datatype: int | None = None,
) -> list[dict[str, Any]]:
    """Extract polygon vertices and metrics in micrometers from a GDS file."""
    kdb = _require_klayout()
    layout = kdb.Layout()
    layout.read(str(path))
    top_cell = layout.top_cell()
    if top_cell is None:
        raise ValueError(f"GDS file has no top cell: {path}")

    polygons = []
    dbu_um = float(layout.dbu)
    for layer_index in layout.layer_indices():
        info = layout.get_info(layer_index)
        if layer is not None and info.layer != layer:
            continue
        if datatype is not None and info.datatype != datatype:
            continue
        region = kdb.Region(top_cell.begin_shapes_rec(layer_index))
        for index, polygon in enumerate(region.each()):
            polygons.append(
                {
                    "layer": int(info.layer),
                    "datatype": int(info.datatype),
                    "polygon_index": index,
                    "points_um": [
                        {"x": point.x * dbu_um, "y": point.y * dbu_um} for point in polygon.each_point_hull()
                    ],
                    "area_um2": float(polygon.area() * dbu_um**2),
                    "bbox_um": _bbox_um(polygon.bbox(), dbu_um),
                }
            )
    return polygons


def build_layout_record(
    path: str | Path,
    *,
    component_name: str,
    gds_path: str,
    source_id: str | None = None,
    design_options: dict[str, Any] | None = None,
    simulation_config: str | None = None,
    campaign: str | None = None,
) -> dict[str, Any]:
    """Build one layout-manifest row from a GDS artifact and its provenance."""
    artifact_path = Path(path)
    summary = parse_gds_summary(artifact_path)
    return {
        "layout_id": summary.pop("layout_id"),
        "artifact_id": f"sha256:{sha256_file(artifact_path)}",
        "design_id": canonical_design_id(component_name, design_options) if design_options else None,
        "component": "coupler",
        "component_name": component_name,
        "artifact_format": "gds",
        "gds_path": gds_path,
        "size_bytes": artifact_path.stat().st_size,
        "source_id": source_id,
        "campaign": campaign,
        "simulation_repo": "SQuADDS/SQuADDS_DB" if simulation_config else None,
        "simulation_config": simulation_config,
        "has_simulation": simulation_config is not None,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        **summary,
    }


def write_manifest(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write a deterministic, viewer-friendly Parquet manifest."""
    if not records:
        raise ValueError("Cannot write an empty layout manifest.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("gds_path").to_parquet(output, index=False)
