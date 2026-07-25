"""Lazy Hugging Face access to SQuADDS layout artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from .manifest import DEFAULT_LAYOUT_REPOSITORY, parse_gds_polygons, parse_gds_summary, sha256_file


@dataclass(frozen=True)
class LayoutReference:
    """A manifest-backed pointer to an immutable layout artifact."""

    layout_id: str
    artifact_id: str
    gds_path: str
    component_name: str
    source_id: str | None = None
    design_id: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> LayoutReference:
        return cls(
            layout_id=record["layout_id"],
            artifact_id=record["artifact_id"],
            gds_path=record["gds_path"],
            component_name=record["component_name"],
            source_id=record.get("source_id"),
            design_id=record.get("design_id"),
        )


class LayoutClient:
    """Find, download, and inspect GDS files without eager bulk downloads."""

    def __init__(
        self,
        repo_id: str = DEFAULT_LAYOUT_REPOSITORY,
        revision: str = "main",
        manifest_filename: str = "metadata/manifest.parquet",
        geometry_features_filename: str = "metadata/geometry-features-v1.parquet",
        manifest_path: str | Path | None = None,
        geometry_features_path: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ):
        self.repo_id = repo_id
        self.revision = revision
        self.manifest_filename = manifest_filename
        self.geometry_features_filename = geometry_features_filename
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.geometry_features_path = Path(geometry_features_path) if geometry_features_path else None
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self._manifest: pd.DataFrame | None = None
        self._geometry_features: pd.DataFrame | None = None

    def manifest(self) -> pd.DataFrame:
        """Load the compact manifest only; raw GDS files remain remote."""
        if self._manifest is None:
            path = self.manifest_path or Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename=self.manifest_filename,
                    revision=self.revision,
                )
            )
            self._manifest = pd.read_parquet(path)
        return self._manifest.copy()

    def find(
        self,
        *,
        layout_id: str | None = None,
        design_id: str | None = None,
        source_id: str | None = None,
    ) -> LayoutReference:
        """Resolve exactly one artifact by its stable identity or provenance."""
        supplied = {"layout_id": layout_id, "design_id": design_id, "source_id": source_id}
        filters = {key: value for key, value in supplied.items() if value is not None}
        if len(filters) != 1:
            raise ValueError("Provide exactly one of layout_id, design_id, or source_id.")
        key, value = next(iter(filters.items()))
        matches = self.manifest().loc[lambda frame: frame[key] == value]
        if len(matches) == 0:
            raise LookupError(f"No layout found for {key}={value!r}.")
        if len(matches) > 1:
            raise LookupError(f"Multiple layouts found for {key}={value!r}; use layout_id instead.")
        return LayoutReference.from_record(matches.iloc[0].to_dict())

    def geometry_features(self, reference: LayoutReference) -> dict[str, Any]:
        """Return the versioned numerical geometry features for one layout."""
        if self._geometry_features is None:
            path = self.geometry_features_path or Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename=self.geometry_features_filename,
                    revision=self.revision,
                )
            )
            self._geometry_features = pd.read_parquet(path)
        matches = self._geometry_features.loc[
            lambda frame: frame["layout_id"] == reference.layout_id
        ]
        if len(matches) != 1:
            raise LookupError(f"No unique geometry feature record for {reference.layout_id!r}.")
        return matches.iloc[0].to_dict()

    def download(self, reference: LayoutReference, verify_checksum: bool = True) -> Path:
        """Download one GDS artifact and verify its immutable content checksum."""
        path = (
            self.artifact_root / reference.gds_path
            if self.artifact_root
            else Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename=reference.gds_path,
                    revision=self.revision,
                )
            )
        )
        if not path.is_file():
            raise FileNotFoundError(f"Layout artifact is missing: {path}")
        if verify_checksum and sha256_file(path) != reference.artifact_id.removeprefix("sha256:"):
            raise ValueError(f"Checksum mismatch for {reference.gds_path}.")
        return path

    def summary(self, reference: LayoutReference) -> dict[str, Any]:
        """Download and inspect a single GDS file using the optional GDS backend."""
        return parse_gds_summary(self.download(reference))

    def polygons(
        self,
        reference: LayoutReference,
        *,
        layer: int | None = None,
        datatype: int | None = None,
    ) -> list[dict[str, Any]]:
        """Download and return selected polygon vertices in micrometers."""
        return parse_gds_polygons(self.download(reference), layer=layer, datatype=datatype)
