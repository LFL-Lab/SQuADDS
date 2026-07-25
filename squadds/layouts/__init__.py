"""Versioned layout artifacts and geometry inspection for SQuADDS."""

from .client import LayoutClient, LayoutReference
from .embeddings import GeometryEmbeddingClient, build_geometry_vectors, geometry_vector_schema
from .features import GEOMETRY_FEATURE_SCHEMA_VERSION, build_geometry_features, geometry_feature_record
from .manifest import (
    DEFAULT_LAYOUT_REPOSITORY,
    build_layout_record,
    canonical_design_id,
    parse_gds_polygons,
    parse_gds_summary,
    write_manifest,
)

__all__ = [
    "DEFAULT_LAYOUT_REPOSITORY",
    "GeometryEmbeddingClient",
    "GEOMETRY_FEATURE_SCHEMA_VERSION",
    "LayoutClient",
    "LayoutReference",
    "build_layout_record",
    "build_geometry_vectors",
    "build_geometry_features",
    "canonical_design_id",
    "geometry_feature_record",
    "geometry_vector_schema",
    "parse_gds_polygons",
    "parse_gds_summary",
    "write_manifest",
]
