#!/usr/bin/env python
"""Stage the unified-convention layout release for SQuADDS_Layouts.

Merges the regenerated ``TransmonCross`` and ``CapNInterdigitalTee`` rows into
the published manifest, keeping the untouched ``GeneralizedCapNInterdigital`` and
``CavityClawRouteMeander`` rows exactly as published, and rebuilds the geometry
feature table from the merged result.

``layout_id`` is a content hash, so it necessarily changes for the two
regenerated families.  ``design_id`` and ``source_id`` do not, and those are the
keys the database bridge and every published study join on.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from squadds.layouts import build_geometry_features
from squadds.layouts.layer_semantics import LAYER_SEMANTICS

REGENERATED = ("TransmonCross", "CapNInterdigitalTee")
PRESERVED = ("GeneralizedCapNInterdigital", "CavityClawRouteMeander")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="port-complete layout-dataset directory")
    parser.add_argument("staging_dir", type=Path)
    parser.add_argument("--published-revision", default="main")
    arguments = parser.parse_args()

    published_manifest = pd.read_parquet(
        hf_hub_download(
            "SQuADDS/SQuADDS_Layouts",
            "metadata/manifest.parquet",
            repo_type="dataset",
            revision=arguments.published_revision,
        )
    )
    regenerated = pd.read_parquet(arguments.dataset_root / "metadata/manifest.parquet")
    if list(published_manifest.columns) != list(regenerated.columns):
        raise SystemExit("manifest schema drift between the published and regenerated tables")

    preserved = published_manifest[published_manifest.component_name.isin(PRESERVED)]
    incoming = regenerated[regenerated.component_name.isin(REGENERATED)]
    merged = pd.concat([preserved, incoming], ignore_index=True).sort_values("gds_path").reset_index(drop=True)

    expected = published_manifest.component_name.value_counts().to_dict()
    actual = merged.component_name.value_counts().to_dict()
    if expected != actual:
        raise SystemExit(f"row counts changed: published {expected} vs merged {actual}")

    # design_id and source_id are the stable keys; assert they survive.
    for column in ("design_id", "source_id"):
        before = set(published_manifest[column].dropna())
        after = set(merged[column].dropna())
        if before != after:
            raise SystemExit(f"{column} set changed; it must be stable across releases")

    staging = arguments.staging_dir
    metadata = staging / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(metadata / "manifest.parquet", index=False)
    build_geometry_features(merged).to_parquet(metadata / "geometry-features-v1.parquet", index=False)
    (metadata / "layer-semantics-v1.json").write_text(json.dumps(LAYER_SEMANTICS, indent=2) + "\n")

    copied = 0
    for row in incoming.itertuples():
        source = arguments.dataset_root / row.gds_path
        destination = staging / row.gds_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

    changed = len(set(merged.layout_id) - set(published_manifest.layout_id))
    summary = {
        "rows": int(len(merged)),
        "regenerated_families": list(REGENERATED),
        "preserved_families": list(PRESERVED),
        "gds_files_staged": copied,
        "layout_ids_changed": changed,
        "design_id_stable": True,
        "source_id_stable": True,
        "layer_semantics_schema_version": LAYER_SEMANTICS["schema_version"],
        "per_family": merged.component_name.value_counts().to_dict(),
    }
    (staging / "release-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
