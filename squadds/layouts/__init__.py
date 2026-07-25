"""Versioned layout artifacts and geometry inspection for SQuADDS."""

from .client import LayoutClient, LayoutReference
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
    "LayoutClient",
    "LayoutReference",
    "build_layout_record",
    "canonical_design_id",
    "parse_gds_polygons",
    "parse_gds_summary",
    "write_manifest",
]
