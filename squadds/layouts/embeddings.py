"""Static v0 layout embeddings with parameters, moments, and shape pixels."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .manifest import parse_gds_polygons

DEFAULT_EMBEDDING_REPOSITORY = "SQuADDS/SQuADDS_Layout_Embeddings"
STATIC_EMBEDDING_MODEL = "static-shape-v0"
STATIC_EMBEDDING_SCHEMA_VERSION = "0.1.0"
SHAPE_SIZE = 96
GEOMETRIC_MOMENT_NAMES = [
    "log1p_functional_area_um2",
    "log1p_functional_perimeter_um",
    "log_functional_bbox_aspect_ratio",
    "bitmap_occupancy_fraction",
    "bitmap_centroid_x",
    "bitmap_centroid_y",
    "bitmap_mu20",
    "bitmap_mu02",
    "bitmap_mu11",
    "bitmap_eccentricity",
]
PARAMETER_BLOCK_SIZE = 1
MOMENT_BLOCK_SIZE = len(GEOMETRIC_MOMENT_NAMES)
SHAPE_BLOCK_SIZE = SHAPE_SIZE * SHAPE_SIZE
EMBEDDING_DIMENSIONS = PARAMETER_BLOCK_SIZE + MOMENT_BLOCK_SIZE + SHAPE_BLOCK_SIZE

_NUMBER_WITH_UNIT = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµμ]*)\s*$")
_UNIT_TO_UM = {
    "": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    "deg": 1.0,
}


def _numeric_parameter_values(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        yield float(value)
    elif isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, str):
        match = _NUMBER_WITH_UNIT.match(value)
        if match and match.group(2) in _UNIT_TO_UM:
            yield float(match.group(1)) * _UNIT_TO_UM[match.group(2)]
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _numeric_parameter_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _numeric_parameter_values(nested)


def parameter_sum(design_options: dict[str, Any]) -> float:
    """Apply the v0 permutation- and dimension-invariant sum operation."""
    return float(sum(_numeric_parameter_values(design_options)))


def _functional_role(component_name: str, layer: int, datatype: int) -> str | None:
    """Map GDS layers to the single signed bitmap used by static-shape-v0."""
    if (layer, datatype) == (1, 10):
        return "conductor"
    if component_name == "CapNInterdigitalTee" and (layer, datatype) == (1, 11):
        return "etch"
    if component_name == "GeneralizedCapNInterdigital" and layer in {2, 3}:
        return "port"
    return None


def _polygon_perimeter(points: list[dict[str, float]]) -> float:
    perimeter = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        perimeter += math.hypot(second["x"] - first["x"], second["y"] - first["y"])
    return perimeter


def _draw_mask(
    polygons: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    size: int,
    supersample: int = 4,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Shape rasterization requires the optional dependency group: uv sync --extra gds") from exc

    left, bottom, right, top = bounds
    width = max(right - left, 1e-12)
    height = max(top - bottom, 1e-12)
    usable = size - 8
    scale = min(usable / width, usable / height)
    offset_x = (size - width * scale) / 2
    offset_y = (size - height * scale) / 2
    canvas_size = size * supersample
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [
            (
                (offset_x + (point["x"] - left) * scale) * supersample,
                (size - offset_y - (point["y"] - bottom) * scale) * supersample,
            )
            for point in polygon["points_um"]
        ]
        draw.polygon(points, fill=255)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(image.resize((size, size), resampling), dtype=np.float32) / 255.0


def rasterize_functional_shape(
    path: str | Path,
    component_name: str,
    size: int = SHAPE_SIZE,
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    """Rasterize functional geometry and calculate the ten v0 moments."""
    polygons = parse_gds_polygons(path)
    by_role: dict[str, list[dict[str, Any]]] = {"conductor": [], "etch": [], "port": []}
    functional = []
    for polygon in polygons:
        role = _functional_role(component_name, polygon["layer"], polygon["datatype"])
        if role:
            by_role[role].append(polygon)
            functional.append(polygon)
    if not functional:
        raise ValueError(f"No functional geometry layers found in {path} for {component_name}.")

    left = min(point["x"] for polygon in functional for point in polygon["points_um"])
    bottom = min(point["y"] for polygon in functional for point in polygon["points_um"])
    right = max(point["x"] for polygon in functional for point in polygon["points_um"])
    top = max(point["y"] for polygon in functional for point in polygon["points_um"])
    bounds = (left, bottom, right, top)

    conductor = _draw_mask(by_role["conductor"], bounds, size)
    etch = _draw_mask(by_role["etch"], bounds, size)
    ports = _draw_mask(by_role["port"], bounds, size)
    bitmap = np.clip(conductor - etch + 0.5 * ports, -1.0, 1.0).astype(np.float32)

    weights = np.abs(bitmap)
    occupancy = weights > 1e-6
    occupancy_fraction = float(np.mean(occupancy))
    yy, xx = np.indices(bitmap.shape, dtype=np.float64)
    total_weight = max(float(weights.sum()), 1e-12)
    centroid_x_pixel = float((xx * weights).sum() / total_weight)
    centroid_y_pixel = float((yy * weights).sum() / total_weight)
    normalized_x = (xx - centroid_x_pixel) / max(size - 1, 1)
    normalized_y = (yy - centroid_y_pixel) / max(size - 1, 1)
    mu20 = float((normalized_x**2 * weights).sum() / total_weight)
    mu02 = float((normalized_y**2 * weights).sum() / total_weight)
    mu11 = float((normalized_x * normalized_y * weights).sum() / total_weight)
    eigenvalues = np.linalg.eigvalsh(np.asarray([[mu20, mu11], [mu11, mu02]]))
    eccentricity = float(math.sqrt(max(0.0, 1.0 - eigenvalues[0] / max(eigenvalues[1], 1e-12))))

    functional_area = sum(float(polygon["area_um2"]) for polygon in functional)
    functional_perimeter = sum(_polygon_perimeter(polygon["points_um"]) for polygon in functional)
    bbox_width = max(right - left, 1e-12)
    bbox_height = max(top - bottom, 1e-12)
    moments = [
        math.log1p(functional_area),
        math.log1p(functional_perimeter),
        math.log(bbox_width / bbox_height),
        occupancy_fraction,
        2.0 * centroid_x_pixel / max(size - 1, 1) - 1.0,
        2.0 * centroid_y_pixel / max(size - 1, 1) - 1.0,
        mu20,
        mu02,
        mu11,
        eccentricity,
    ]
    metadata = {
        "functional_bounds_um": {"left": left, "bottom": bottom, "right": right, "top": top},
        "functional_layers": [
            {"layer": polygon["layer"], "datatype": polygon["datatype"], "role": role}
            for role, role_polygons in by_role.items()
            for polygon in role_polygons[:1]
        ],
    }
    return bitmap, moments, metadata


def static_embedding_schema(
    parameter_mean: float, parameter_std: float, moment_mean: np.ndarray, moment_std: np.ndarray
) -> dict[str, Any]:
    """Describe every dimension and normalization operation in static-shape-v0."""
    return {
        "model": STATIC_EMBEDDING_MODEL,
        "embedding_schema_version": STATIC_EMBEDDING_SCHEMA_VERSION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "blocks": {
            "parameter_sum": {"offset": 0, "dimensions": 1},
            "geometric_moments": {
                "offset": 1,
                "dimensions": MOMENT_BLOCK_SIZE,
                "names": GEOMETRIC_MOMENT_NAMES,
            },
            "shape_bitmap": {
                "offset": 1 + MOMENT_BLOCK_SIZE,
                "dimensions": SHAPE_BLOCK_SIZE,
                "shape": [SHAPE_SIZE, SHAPE_SIZE],
                "flatten_order": "row-major",
            },
        },
        "shape_rasterization": {
            "resolution": [SHAPE_SIZE, SHAPE_SIZE],
            "crop": "functional geometry bounds, aspect preserving, centered with 4-pixel margin",
            "supersampling": 4,
            "values": {"conductor": 1.0, "etch": -1.0, "port": 0.5, "background": 0.0},
            "excluded": "simulation-domain ground layer (1, 0)",
        },
        "normalization": {
            "parameter_mean": parameter_mean,
            "parameter_std": parameter_std,
            "moment_mean": moment_mean.tolist(),
            "moment_std": moment_std.tolist(),
            "parameter": "tanh(z-score)",
            "moments": "z-score, clipped to [-5, 5], then L2 normalized",
            "shape": "signed bitmap, then L2 normalized",
            "embedding": "concatenated blocks, then L2 normalized",
        },
    }


def build_static_embeddings(
    manifest: pd.DataFrame,
    design_options_by_id: dict[str, dict[str, Any]],
    artifact_resolver: Callable[[dict[str, Any]], Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build screenshot-specified static v0 embeddings for all manifest rows."""
    raw_records = []
    for record in manifest.to_dict(orient="records"):
        design_id = record.get("design_id")
        if design_id not in design_options_by_id:
            raise LookupError(f"No design options found for {design_id!r}.")
        path = artifact_resolver(record)
        bitmap, moments, shape_metadata = rasterize_functional_shape(path, record["component_name"])
        raw_records.append(
            {
                "record": record,
                "parameter_sum": parameter_sum(design_options_by_id[design_id]),
                "moments": np.asarray(moments, dtype=np.float64),
                "bitmap": bitmap.reshape(-1).astype(np.float64),
                "shape_metadata": shape_metadata,
            }
        )

    parameter_values = np.asarray([item["parameter_sum"] for item in raw_records], dtype=np.float64)
    moment_values = np.vstack([item["moments"] for item in raw_records])
    parameter_mean = float(parameter_values.mean())
    parameter_std = float(parameter_values.std()) or 1.0
    moment_mean = moment_values.mean(axis=0)
    moment_std = moment_values.std(axis=0)
    moment_std[moment_std == 0] = 1.0
    schema = static_embedding_schema(parameter_mean, parameter_std, moment_mean, moment_std)

    embeddings = []
    for item in raw_records:
        parameter_block = np.asarray([math.tanh((item["parameter_sum"] - parameter_mean) / parameter_std)])
        moment_block = np.clip((item["moments"] - moment_mean) / moment_std, -5.0, 5.0)
        moment_norm = np.linalg.norm(moment_block)
        if moment_norm:
            moment_block /= moment_norm
        shape_block = item["bitmap"]
        shape_norm = np.linalg.norm(shape_block)
        if shape_norm:
            shape_block /= shape_norm
        embedding = np.concatenate([parameter_block, moment_block, shape_block])
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm:
            embedding /= embedding_norm
        record = item["record"]
        embeddings.append(
            {
                "layout_id": record["layout_id"],
                "artifact_id": record["artifact_id"],
                "design_id": record.get("design_id"),
                "component_name": record["component_name"],
                "source_id": record.get("source_id"),
                "embedding_model": STATIC_EMBEDDING_MODEL,
                "embedding_schema_version": STATIC_EMBEDDING_SCHEMA_VERSION,
                "parameter_sum": item["parameter_sum"],
                "geometric_moments": item["moments"].astype(np.float32).tolist(),
                "shape_bitmap_sha256": hashlib.sha256(item["bitmap"].astype(np.float32).tobytes()).hexdigest(),
                "functional_bounds_um": item["shape_metadata"]["functional_bounds_um"],
                "embedding": embedding.astype(np.float32).tolist(),
            }
        )
    return pd.DataFrame(embeddings), schema


class StaticEmbeddingClient:
    """Load static v0 vectors lazily and search by cosine similarity."""

    def __init__(
        self,
        repo_id: str = DEFAULT_EMBEDDING_REPOSITORY,
        revision: str = "main",
        filename: str = "metadata/static-embedding-v0.parquet",
        embedding_path: str | Path | None = None,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.embedding_path = Path(embedding_path) if embedding_path else None
        self._embeddings: pd.DataFrame | None = None

    def embeddings(self) -> pd.DataFrame:
        """Load the vector table; raw GDS artifacts remain remote."""
        if self._embeddings is None:
            path = self.embedding_path or Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename=self.filename,
                    revision=self.revision,
                )
            )
            self._embeddings = pd.read_parquet(path)
        return self._embeddings.copy()

    def get(self, layout_id: str) -> dict[str, Any]:
        """Return one static v0 embedding record by stable layout identity."""
        matches = self.embeddings().loc[lambda frame: frame["layout_id"] == layout_id]
        if len(matches) != 1:
            raise LookupError(f"No unique embedding record for {layout_id!r}.")
        return matches.iloc[0].to_dict()

    def shape_bitmap(self, layout_id: str) -> np.ndarray:
        """Recover the 96x96 signed shape bitmap from a v0 embedding."""
        embedding = np.asarray(self.get(layout_id)["embedding"], dtype=np.float32)
        return embedding[PARAMETER_BLOCK_SIZE + MOMENT_BLOCK_SIZE :].reshape(SHAPE_SIZE, SHAPE_SIZE)

    def nearest(
        self,
        layout_id: str,
        limit: int = 10,
        component_name: str | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """Return cosine-nearest v0 layouts, optionally within one component family."""
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100.")
        frame = self.embeddings()
        query = self.get(layout_id)
        candidates = frame.loc[frame["layout_id"] != layout_id].copy()
        if component_name is not None:
            candidates = candidates.loc[candidates["component_name"] == component_name]
        if candidates.empty:
            return []
        matrix = np.vstack(candidates["embedding"].map(lambda value: np.asarray(value, dtype=np.float32)))
        similarities = matrix @ np.asarray(query["embedding"], dtype=np.float32)
        candidates["cosine_similarity"] = similarities
        result = candidates.nlargest(limit, "cosine_similarity")
        if not include_embeddings:
            result = result.drop(columns=["embedding"])
        return result.to_dict(orient="records")


def write_static_embedding_dataset(
    manifest: pd.DataFrame,
    design_options_by_id: dict[str, dict[str, Any]],
    artifact_resolver: Callable[[dict[str, Any]], Path],
    output_dir: str | Path,
) -> tuple[int, int]:
    """Write static v0 vectors and schema JSON for a Hugging Face release."""
    output = Path(output_dir)
    vectors, schema = build_static_embeddings(manifest, design_options_by_id, artifact_resolver)
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    vectors.to_parquet(metadata / "static-embedding-v0.parquet", index=False)
    (metadata / "static-embedding-v0.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return len(vectors), schema["dimensions"]
