"""Deterministic geometry-vector embeddings and nearest-layout lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

DEFAULT_EMBEDDING_REPOSITORY = "SQuADDS/SQuADDS_Layout_Embeddings"
GEOMETRY_VECTOR_SCHEMA_VERSION = "1.0.0"


def _layer_vocabulary(features: pd.DataFrame) -> list[tuple[int, int]]:
    """Collect a stable sorted vocabulary of layer/datatype pairs."""
    pairs = set()
    for layer_records in features["layer_features"]:
        for record in layer_records:
            pairs.add((int(record["layer"]), int(record["datatype"])))
    return sorted(pairs)


def geometry_vector_schema(features: pd.DataFrame) -> dict[str, Any]:
    """Describe the fixed vector layout derived from a geometry-feature table."""
    layer_pairs = _layer_vocabulary(features)
    feature_names = [
        "log1p_bbox_width_um",
        "log1p_bbox_height_um",
        "log1p_bbox_area_um2",
        "log_bbox_aspect_ratio",
        "log1p_total_area_um2",
        "log1p_polygon_count",
        "log1p_cell_count",
        "log1p_layer_count",
    ]
    for layer, datatype in layer_pairs:
        prefix = f"layer_{layer}_datatype_{datatype}"
        feature_names.extend([f"{prefix}_present", f"{prefix}_area_fraction", f"{prefix}_polygon_fraction"])
    return {
        "geometry_vector_schema_version": GEOMETRY_VECTOR_SCHEMA_VERSION,
        "source_geometry_feature_schema_version": "1.0.0",
        "normalization": "L2 unit norm; raw dimensions use log1p except log aspect ratio and layer fractions.",
        "layer_vocabulary": [{"layer": layer, "datatype": datatype} for layer, datatype in layer_pairs],
        "feature_names": feature_names,
        "dimensions": len(feature_names),
    }


def _vector_for_record(record: dict[str, Any], layer_pairs: list[tuple[int, int]]) -> np.ndarray:
    aspect_ratio = max(float(record.get("bbox_aspect_ratio") or 1.0), 1e-12)
    vector = [
        np.log1p(float(record.get("bbox_width_um", 0.0))),
        np.log1p(float(record.get("bbox_height_um", 0.0))),
        np.log1p(float(record.get("bbox_area_um2", 0.0))),
        np.log(aspect_ratio),
        np.log1p(float(record.get("total_area_um2", 0.0))),
        np.log1p(float(record.get("polygon_count", 0))),
        np.log1p(float(record.get("cell_count", 0))),
        np.log1p(float(record.get("layer_count", 0))),
    ]
    layer_records = {(int(item["layer"]), int(item["datatype"])): item for item in record["layer_features"]}
    total_area = max(float(record.get("total_area_um2", 0.0)), 1e-12)
    total_polygons = max(float(record.get("polygon_count", 0.0)), 1.0)
    for pair in layer_pairs:
        item = layer_records.get(pair)
        if item is None:
            vector.extend([0.0, 0.0, 0.0])
        else:
            vector.extend(
                [
                    1.0,
                    float(item["area_um2"]) / total_area,
                    float(item["polygon_count"]) / total_polygons,
                ]
            )
    vector_array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector_array)
    return vector_array if norm == 0 else vector_array / norm


def build_geometry_vectors(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one reproducible unit vector per layout from geometry features."""
    schema = geometry_vector_schema(features)
    layer_pairs = [(item["layer"], item["datatype"]) for item in schema["layer_vocabulary"]]
    records = []
    for record in features.to_dict(orient="records"):
        vector = _vector_for_record(record, layer_pairs)
        records.append(
            {
                "layout_id": record["layout_id"],
                "artifact_id": record["artifact_id"],
                "design_id": record.get("design_id"),
                "component_name": record["component_name"],
                "source_id": record.get("source_id"),
                "embedding_schema_version": GEOMETRY_VECTOR_SCHEMA_VERSION,
                "embedding": vector.tolist(),
            }
        )
    return pd.DataFrame(records), schema


class GeometryEmbeddingClient:
    """Load geometry vectors lazily and search by cosine similarity."""

    def __init__(
        self,
        repo_id: str = DEFAULT_EMBEDDING_REPOSITORY,
        revision: str = "main",
        filename: str = "metadata/geometry-vector-v1.parquet",
        embedding_path: str | Path | None = None,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self.embedding_path = Path(embedding_path) if embedding_path else None
        self._embeddings: pd.DataFrame | None = None

    def embeddings(self) -> pd.DataFrame:
        """Load the compact vector table; raw GDS artifacts remain remote."""
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
        """Return one embedding record by stable layout identity."""
        matches = self.embeddings().loc[lambda frame: frame["layout_id"] == layout_id]
        if len(matches) != 1:
            raise LookupError(f"No unique embedding record for {layout_id!r}.")
        return matches.iloc[0].to_dict()

    def nearest(self, layout_id: str, limit: int = 10, component_name: str | None = None) -> list[dict[str, Any]]:
        """Return cosine-nearest layouts, optionally within one component family."""
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100.")
        frame = self.embeddings()
        query = self.get(layout_id)
        candidates = frame.loc[frame["layout_id"] != layout_id].copy()
        if component_name is not None:
            candidates = candidates.loc[candidates["component_name"] == component_name]
        matrix = np.vstack(candidates["embedding"].map(np.asarray))
        similarities = matrix @ np.asarray(query["embedding"])
        candidates["cosine_similarity"] = similarities
        return candidates.nlargest(limit, "cosine_similarity").to_dict(orient="records")


def write_geometry_vector_dataset(features: pd.DataFrame, output_dir: str | Path) -> tuple[int, int]:
    """Write the vector table and schema JSON for a Hugging Face dataset release."""
    output = Path(output_dir)
    vectors, schema = build_geometry_vectors(features)
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    vectors.to_parquet(metadata / "geometry-vector-v1.parquet", index=False)
    (metadata / "geometry-vector-v1.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return len(vectors), schema["dimensions"]
