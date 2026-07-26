"""Versioned layout artifacts and geometry inspection for SQuADDS."""

from .client import LayoutClient, LayoutReference
from .embeddings import (
    EMBEDDING_DIMENSIONS,
    SHAPE_SIZE,
    STATIC_EMBEDDING_MODEL,
    StaticEmbeddingClient,
    build_static_embeddings,
    parameter_sum,
    rasterize_functional_shape,
)
from .features import GEOMETRY_FEATURE_SCHEMA_VERSION, build_geometry_features, geometry_feature_record
from .manifest import (
    DEFAULT_LAYOUT_REPOSITORY,
    build_layout_record,
    canonical_design_id,
    infer_layout_component_name,
    parse_gds_polygons,
    parse_gds_summary,
    write_manifest,
)
from .transfer import (
    TransferRidgeRegressor,
    V0FeatureProjector,
    V0TransferLearningStudy,
    compress_v0_embeddings,
    evaluate_transfer_learning,
    regression_scores,
    required_target_samples,
    summarize_learning_curve,
    target_to_source_similarity,
)

__all__ = [
    "DEFAULT_LAYOUT_REPOSITORY",
    "EMBEDDING_DIMENSIONS",
    "GEOMETRY_FEATURE_SCHEMA_VERSION",
    "LayoutClient",
    "LayoutReference",
    "build_layout_record",
    "SHAPE_SIZE",
    "STATIC_EMBEDDING_MODEL",
    "StaticEmbeddingClient",
    "TransferRidgeRegressor",
    "V0FeatureProjector",
    "V0TransferLearningStudy",
    "build_static_embeddings",
    "build_geometry_features",
    "canonical_design_id",
    "compress_v0_embeddings",
    "evaluate_transfer_learning",
    "infer_layout_component_name",
    "geometry_feature_record",
    "parameter_sum",
    "rasterize_functional_shape",
    "regression_scores",
    "required_target_samples",
    "summarize_learning_curve",
    "target_to_source_similarity",
    "parse_gds_polygons",
    "parse_gds_summary",
    "write_manifest",
]
