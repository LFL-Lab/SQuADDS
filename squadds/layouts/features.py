"""Versioned numerical geometry features derived from layout manifests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

GEOMETRY_FEATURE_SCHEMA_VERSION = "1.0.0"


def geometry_feature_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one manifest record into stable, model-ready geometry features.

    This intentionally preserves per-layer values as sparse records instead of
    selecting a fixed layer vocabulary. A future embedding model can choose a
    vocabulary without changing the source geometry contract.
    """
    bbox = record.get("bbox_um") or {}
    width = float(bbox.get("right", 0.0) - bbox.get("left", 0.0))
    height = float(bbox.get("top", 0.0) - bbox.get("bottom", 0.0))
    layers = record.get("layers")
    if layers is None:
        layers = []
    if not isinstance(layers, Iterable):
        layers = []
    layer_features = sorted(
        (
            {
                "layer": int(entry["layer"]),
                "datatype": int(entry["datatype"]),
                "polygon_count": int(entry["polygon_count"]),
                "area_um2": float(entry["area_um2"]),
            }
            for entry in layers
        ),
        key=lambda entry: (entry["layer"], entry["datatype"]),
    )
    return {
        "layout_id": record["layout_id"],
        "artifact_id": record["artifact_id"],
        "design_id": record.get("design_id"),
        "component_name": record["component_name"],
        "source_id": record.get("source_id"),
        "geometry_feature_schema_version": GEOMETRY_FEATURE_SCHEMA_VERSION,
        "bbox_width_um": width,
        "bbox_height_um": height,
        "bbox_area_um2": width * height,
        "bbox_aspect_ratio": width / height if height else None,
        "polygon_count": int(record.get("polygon_count", 0)),
        "cell_count": int(record.get("cell_count", 0)),
        "layer_count": len(layer_features),
        "total_area_um2": sum(entry["area_um2"] for entry in layer_features),
        "layer_features": layer_features,
    }


def build_geometry_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build geometry feature rows from the manifest without opening GDS files."""
    return pd.DataFrame(geometry_feature_record(record) for record in manifest.to_dict(orient="records"))
