"""Universal v1 layout embeddings derived from GDS geometry and design parameters."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .embeddings import _draw_mask, _functional_role, _polygon_perimeter
from .manifest import parse_gds_polygons

UNIVERSAL_EMBEDDING_MODEL = "universal-geometry-v1"
UNIVERSAL_EMBEDDING_SCHEMA_VERSION = "1.0.0"
UNIVERSAL_SHAPE_SIZE = 64
UNIVERSAL_METRIC_NAMES = [
    "log1p_total_area_um2",
    "log1p_total_perimeter_um",
    "log1p_bbox_width_um",
    "log1p_bbox_height_um",
    "log_bbox_aspect_ratio",
    "total_occupancy",
    "conductor_occupancy",
    "etch_occupancy",
    "port_occupancy",
    "centroid_x",
    "centroid_y",
    "mu20",
    "mu02",
    "mu11",
    "eccentricity",
    "horizontal_symmetry",
    "vertical_symmetry",
    "log1p_polygon_count",
    "log1p_vertex_count",
    "log1p_layer_count",
    "log1p_conductor_area_um2",
    "log1p_etch_area_um2",
    "log1p_port_area_um2",
    "log1p_conductor_perimeter_um",
    "log1p_etch_perimeter_um",
    "log1p_port_perimeter_um",
    "conductor_centroid_x",
    "conductor_centroid_y",
    "etch_centroid_x",
    "etch_centroid_y",
    "port_centroid_x",
    "port_centroid_y",
]
UNIVERSAL_METRIC_VALUE_DIMENSIONS = len(UNIVERSAL_METRIC_NAMES)
UNIVERSAL_METRIC_DIMENSIONS = 2 * UNIVERSAL_METRIC_VALUE_DIMENSIONS
UNIVERSAL_SHAPE_CHANNELS = [
    "conductor_occupancy",
    "etch_occupancy",
    "port_occupancy",
    "signed_functional_material",
    "signed_distance_to_functional_boundary",
]
UNIVERSAL_DCT_SIZE = 8
UNIVERSAL_SHAPE_DIMENSIONS = len(UNIVERSAL_SHAPE_CHANNELS) * UNIVERSAL_DCT_SIZE**2
UNIVERSAL_CONTROL_DIMENSIONS = 128
UNIVERSAL_EMBEDDING_DIMENSIONS = UNIVERSAL_METRIC_DIMENSIONS + UNIVERSAL_SHAPE_DIMENSIONS + UNIVERSAL_CONTROL_DIMENSIONS

_NUMBER_WITH_UNIT = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]*)\s*$")
_UNIT_SCALE = {
    "": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    "deg": math.pi / 180.0,
    "rad": 1.0,
}


def _safe_unit_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


def _mask_moments(mask: np.ndarray) -> tuple[float, float, float, float, float, float]:
    weights = np.abs(mask).astype(np.float64)
    total = float(weights.sum())
    if total <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    yy, xx = np.indices(mask.shape, dtype=np.float64)
    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)
    scale = max(mask.shape[0] - 1, 1)
    nx = (xx - cx) / scale
    ny = (yy - cy) / scale
    mu20 = float((nx**2 * weights).sum() / total)
    mu02 = float((ny**2 * weights).sum() / total)
    mu11 = float((nx * ny * weights).sum() / total)
    eigenvalues = np.linalg.eigvalsh(np.asarray([[mu20, mu11], [mu11, mu02]]))
    eccentricity = math.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[1], 1e-12)))
    return 2 * cx / scale - 1, 2 * cy / scale - 1, mu20, mu02, mu11, eccentricity


def _role_for_polygon(
    component_name: str,
    layer: int,
    datatype: int,
    layer_roles: Mapping[tuple[int, int], str] | None,
) -> str | None:
    if layer_roles is not None:
        return layer_roles.get((layer, datatype))
    role = _functional_role(component_name, layer, datatype)
    if role is not None:
        return role
    # A foreign layout can still be embedded without a SQuADDS component name.
    # Explicit layer-role metadata is preferred; otherwise exclude the common
    # simulation-domain layer and treat remaining geometry as conductor.
    return None if (layer, datatype) == (1, 0) else "conductor"


def _functional_geometry(
    path: str | Path,
    component_name: str,
    layer_roles: Mapping[tuple[int, int], str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], tuple[float, float, float, float]]:
    by_role: dict[str, list[dict[str, Any]]] = {"conductor": [], "etch": [], "port": []}
    for polygon in parse_gds_polygons(path):
        role = _role_for_polygon(component_name, polygon["layer"], polygon["datatype"], layer_roles)
        if role in by_role:
            by_role[role].append(polygon)
    functional = [polygon for polygons in by_role.values() for polygon in polygons]
    if not functional:
        raise ValueError(f"No functional geometry found in {path}. Supply layer_roles for this layout.")
    xs = [point["x"] for polygon in functional for point in polygon["points_um"]]
    ys = [point["y"] for polygon in functional for point in polygon["points_um"]]
    return by_role, (min(xs), min(ys), max(xs), max(ys))


def _shape_descriptor(masks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.fft import dctn
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError("universal-geometry-v1 requires scipy.") from exc

    conductor = masks["conductor"]
    etch = masks["etch"]
    port = masks["port"]
    signed = np.clip(conductor - etch + 0.5 * port, -1.0, 1.0)
    occupied = np.abs(signed) > 1e-3
    signed_distance = distance_transform_edt(occupied) - distance_transform_edt(~occupied)
    signed_distance /= max(float(np.max(np.abs(signed_distance))), 1.0)
    channels = np.stack([conductor, etch, port, signed, signed_distance]).astype(np.float32)
    coefficients = []
    for channel in channels:
        transformed = dctn(channel, type=2, norm="ortho")
        coefficients.append(transformed[:UNIVERSAL_DCT_SIZE, :UNIVERSAL_DCT_SIZE].reshape(-1))
    return np.concatenate(coefficients).astype(np.float32), signed.astype(np.float32)


def _raw_geometry(
    path: str | Path,
    component_name: str,
    layer_roles: Mapping[tuple[int, int], str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    by_role, bounds = _functional_geometry(path, component_name, layer_roles)
    masks = {
        role: _draw_mask(polygons, bounds, UNIVERSAL_SHAPE_SIZE)
        if polygons
        else np.zeros((UNIVERSAL_SHAPE_SIZE, UNIVERSAL_SHAPE_SIZE), dtype=np.float32)
        for role, polygons in by_role.items()
    }
    shape, signed = _shape_descriptor(masks)
    left, bottom, right, top = bounds
    width = max(right - left, 1e-12)
    height = max(top - bottom, 1e-12)
    all_polygons = [polygon for polygons in by_role.values() for polygon in polygons]
    role_area = {role: sum(float(polygon["area_um2"]) for polygon in polygons) for role, polygons in by_role.items()}
    role_perimeter = {
        role: sum(_polygon_perimeter(polygon["points_um"]) for polygon in polygons)
        for role, polygons in by_role.items()
    }
    moments = _mask_moments(signed)
    centroids = {role: _mask_moments(mask)[:2] for role, mask in masks.items()}
    metrics = np.asarray(
        [
            math.log1p(sum(role_area.values())),
            math.log1p(sum(role_perimeter.values())),
            math.log1p(width),
            math.log1p(height),
            math.log(width / height),
            float(np.mean(np.abs(signed) > 1e-3)),
            float(np.mean(masks["conductor"] > 1e-3)),
            float(np.mean(masks["etch"] > 1e-3)),
            float(np.mean(masks["port"] > 1e-3)),
            *moments,
            1.0 - float(np.mean(np.abs(signed - np.fliplr(signed)))) / 2.0,
            1.0 - float(np.mean(np.abs(signed - np.flipud(signed)))) / 2.0,
            math.log1p(len(all_polygons)),
            math.log1p(sum(len(polygon["points_um"]) for polygon in all_polygons)),
            math.log1p(len({(polygon["layer"], polygon["datatype"]) for polygon in all_polygons})),
            *(math.log1p(role_area[role]) for role in ("conductor", "etch", "port")),
            *(math.log1p(role_perimeter[role]) for role in ("conductor", "etch", "port")),
            *(coordinate for role in ("conductor", "etch", "port") for coordinate in centroids[role]),
        ],
        dtype=np.float32,
    )
    available = np.ones_like(metrics)
    for offset, role in zip((20, 21, 22), ("conductor", "etch", "port")):
        if not by_role[role]:
            available[offset] = 0
            available[offset + 3] = 0
            centroid_offset = 26 + 2 * ("conductor", "etch", "port").index(role)
            available[centroid_offset : centroid_offset + 2] = 0
    metadata = {
        "left": float(left),
        "bottom": float(bottom),
        "right": float(right),
        "top": float(top),
    }
    return metrics, available, shape, metadata


def _parameter_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, bool):
        yield prefix, float(value)
    elif isinstance(value, (int, float)):
        yield prefix, float(value)
    elif isinstance(value, str):
        match = _NUMBER_WITH_UNIT.match(value)
        if match and match.group(2) in _UNIT_SCALE:
            yield prefix, float(match.group(1)) * _UNIT_SCALE[match.group(2)]
    elif isinstance(value, dict):
        for key in sorted(value):
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _parameter_scalars(value[key], nested_prefix)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _parameter_scalars(nested, f"{prefix}[{index}]")


def parameter_channels(
    design_options: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[float], list[int], list[int]]:
    """Encode named layout controls without depending on a design-tool schema."""
    vector = np.zeros(UNIVERSAL_CONTROL_DIMENSIONS, dtype=np.float32)
    names, values, indices, signs = [], [], [], []
    for name, value in _parameter_scalars(design_options):
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % UNIVERSAL_CONTROL_DIMENSIONS
        sign = 1 if digest[8] & 1 else -1
        bounded = math.tanh(math.copysign(math.log1p(abs(value)), value) / 4.0)
        vector[index] += sign * bounded
        names.append(name)
        values.append(float(value))
        indices.append(index)
        signs.append(sign)
    return vector, names, values, indices, signs


def universal_embedding_schema(metric_mean: np.ndarray, metric_std: np.ndarray) -> dict[str, Any]:
    """Return the frozen, machine-readable universal-geometry-v1 contract."""
    return {
        "model": UNIVERSAL_EMBEDDING_MODEL,
        "embedding_schema_version": UNIVERSAL_EMBEDDING_SCHEMA_VERSION,
        "dimensions": UNIVERSAL_EMBEDDING_DIMENSIONS,
        "input_contract": {
            "required": ["GDSII geometry", "layout-tool design parameter mapping"],
            "layer_roles": ["conductor", "etch", "port"],
            "foreign_layouts": "provide a (layer, datatype) to role mapping; unspecified non-domain layers fall back to conductor",
            "simulation_results_used": False,
        },
        "blocks": {
            "geometry_metrics": {
                "offset": 0,
                "dimensions": UNIVERSAL_METRIC_DIMENSIONS,
                "values": UNIVERSAL_METRIC_NAMES,
                "availability_mask_offset": UNIVERSAL_METRIC_VALUE_DIMENSIONS,
            },
            "multiscale_shape": {
                "offset": UNIVERSAL_METRIC_DIMENSIONS,
                "dimensions": UNIVERSAL_SHAPE_DIMENSIONS,
                "channels": UNIVERSAL_SHAPE_CHANNELS,
                "raster_resolution": [UNIVERSAL_SHAPE_SIZE, UNIVERSAL_SHAPE_SIZE],
                "transform": f"orthonormal 2D DCT, lowest {UNIVERSAL_DCT_SIZE}x{UNIVERSAL_DCT_SIZE} frequencies",
            },
            "parameter_controls": {
                "offset": UNIVERSAL_METRIC_DIMENSIONS + UNIVERSAL_SHAPE_DIMENSIONS,
                "dimensions": UNIVERSAL_CONTROL_DIMENSIONS,
                "transform": "signed SHA-256 feature hashing of canonical parameter paths and bounded unit-normalized values",
            },
        },
        "normalization": {
            "metric_mean": metric_mean.astype(float).tolist(),
            "metric_std": metric_std.astype(float).tolist(),
            "metrics": "available values z-scored and clipped to [-6, 6], then concatenated with availability mask",
            "shape": "each DCT channel L2 normalized",
            "controls": "signed-log bounded values accumulated by stable parameter-name hash",
            "blocks": "each block L2 normalized; concatenation divided by sqrt(3)",
        },
        "invariances": {
            "translation": "geometry is cropped to functional bounds and centered",
            "scale": "shape block is scale invariant; metric block retains physical dimensions",
            "parameter_order": "sorted traversal and commutative hashed accumulation",
        },
    }


def _assemble_embedding(
    metrics: np.ndarray,
    available: np.ndarray,
    shape: np.ndarray,
    controls: np.ndarray,
    metric_mean: np.ndarray,
    metric_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = np.where(available > 0, (metrics - metric_mean) / metric_std, 0.0)
    standardized = np.clip(standardized, -6.0, 6.0)
    metric_block = _safe_unit_vector(np.concatenate([standardized, available]).astype(np.float32))
    shape_channels = shape.reshape(len(UNIVERSAL_SHAPE_CHANNELS), -1)
    shape_block = np.concatenate([_safe_unit_vector(channel) for channel in shape_channels])
    shape_block = _safe_unit_vector(shape_block.astype(np.float32))
    control_block = _safe_unit_vector(controls.astype(np.float32))
    embedding = np.concatenate([metric_block, shape_block, control_block]) / math.sqrt(3.0)
    return _safe_unit_vector(embedding.astype(np.float32)), standardized.astype(np.float32)


def write_universal_embedding_dataset(
    manifest: pd.DataFrame,
    design_options_by_id: dict[str, dict[str, Any]],
    artifact_resolver: Callable[[dict[str, Any]], Path],
    output_dir: str | Path,
    *,
    component_name: str | None = None,
    layer_roles: Mapping[tuple[int, int], str] | None = None,
) -> tuple[int, int]:
    """Generate a streaming v1 Parquet release and its frozen schema."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Writing universal embeddings requires pyarrow.") from exc

    selected = manifest
    if component_name is not None:
        selected = selected.loc[selected["component_name"] == component_name]
    records = selected.to_dict(orient="records")
    if not records:
        raise ValueError("Cannot write an empty embedding catalogue.")

    output = Path(output_dir)
    metadata_dir = output / "metadata"
    model_dir = output / "models" / UNIVERSAL_EMBEDDING_MODEL
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="squadds-v1-") as temporary:
        temporary_path = Path(temporary)
        metrics_store = np.memmap(
            temporary_path / "metrics.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(records), UNIVERSAL_METRIC_VALUE_DIMENSIONS),
        )
        available_store = np.memmap(
            temporary_path / "available.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(records), UNIVERSAL_METRIC_VALUE_DIMENSIONS),
        )
        shape_store = np.memmap(
            temporary_path / "shape.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(records), UNIVERSAL_SHAPE_DIMENSIONS),
        )
        control_store = np.memmap(
            temporary_path / "controls.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(records), UNIVERSAL_CONTROL_DIMENSIONS),
        )
        bounds: list[dict[str, float]] = []
        parameter_metadata = []
        parameter_statistics: dict[str, list[float]] = {}
        for index, record in enumerate(records):
            design_id = record.get("design_id")
            if design_id not in design_options_by_id:
                raise LookupError(f"No design options found for {design_id!r}.")
            metrics, available, shape, geometry_bounds = _raw_geometry(
                artifact_resolver(record), record["component_name"], layer_roles
            )
            controls, names, values, indices, signs = parameter_channels(design_options_by_id[design_id])
            metrics_store[index] = metrics
            available_store[index] = available
            shape_store[index] = shape
            control_store[index] = controls
            bounds.append(geometry_bounds)
            parameter_metadata.append((names, values, indices, signs))
            for name, value in zip(names, values):
                parameter_statistics.setdefault(name, []).append(value)

        available_counts = np.maximum(available_store.sum(axis=0), 1.0)
        metric_mean = np.asarray((metrics_store * available_store).sum(axis=0) / available_counts)
        centered = (metrics_store - metric_mean) * available_store
        metric_std = np.sqrt(np.asarray((centered**2).sum(axis=0) / available_counts))
        metric_std[metric_std < 1e-8] = 1.0
        schema = universal_embedding_schema(metric_mean, metric_std)

        embedding_path = metadata_dir / "universal-geometry-v1.parquet"
        writer = None
        for start in range(0, len(records), 128):
            batch = []
            for index in range(start, min(start + 128, len(records))):
                record = records[index]
                embedding, standardized = _assemble_embedding(
                    metrics_store[index],
                    available_store[index],
                    shape_store[index],
                    control_store[index],
                    metric_mean,
                    metric_std,
                )
                names, values, indices, signs = parameter_metadata[index]
                batch.append(
                    {
                        "layout_id": record["layout_id"],
                        "artifact_id": record["artifact_id"],
                        "design_id": record.get("design_id"),
                        "component_name": record["component_name"],
                        "source_id": record.get("source_id"),
                        "embedding_version": "v1",
                        "embedding_model": UNIVERSAL_EMBEDDING_MODEL,
                        "embedding_schema_version": UNIVERSAL_EMBEDDING_SCHEMA_VERSION,
                        "geometry_metrics": metrics_store[index].tolist(),
                        "standardized_geometry_metrics": standardized.tolist(),
                        "metric_availability": available_store[index].astype(bool).tolist(),
                        "functional_bounds_um": bounds[index],
                        "parameter_names": names,
                        "parameter_values": values,
                        "parameter_hash_indices": indices,
                        "parameter_hash_signs": signs,
                        "shape_descriptor_sha256": hashlib.sha256(shape_store[index].tobytes()).hexdigest(),
                        "embedding": embedding.tolist(),
                    }
                )
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(embedding_path, table.schema, compression="zstd")
            writer.write_table(table)
        assert writer is not None
        writer.close()

    schema_path = model_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    control_rows = []
    for name, values in sorted(parameter_statistics.items()):
        _, _, _, indices, signs = parameter_channels({name: 1.0})
        array = np.asarray(values, dtype=np.float64)
        control_rows.append(
            {
                "parameter_name": name,
                "hash_index": indices[0],
                "hash_sign": signs[0],
                "count": len(values),
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "mean": float(array.mean()),
                "standard_deviation": float(array.std()),
            }
        )
    control_map_path = model_dir / "control-map.parquet"
    pd.DataFrame(control_rows).to_parquet(control_map_path, index=False)
    release_files = {
        embedding_path.relative_to(output).as_posix(): embedding_path,
        schema_path.relative_to(output).as_posix(): schema_path,
        control_map_path.relative_to(output).as_posix(): control_map_path,
    }
    release_manifest = {
        "model": UNIVERSAL_EMBEDDING_MODEL,
        "embedding_schema_version": UNIVERSAL_EMBEDDING_SCHEMA_VERSION,
        "component_name": component_name,
        "rows": len(records),
        "dimensions": schema["dimensions"],
        "files": {
            name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in release_files.items()
        },
    }
    (model_dir / "release-manifest.json").write_text(json.dumps(release_manifest, indent=2) + "\n")
    return len(records), schema["dimensions"]
