#!/usr/bin/env python
"""Stage generated CapNInterdigitalTee GDS files for the layout registry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from squadds.layouts import build_geometry_features, build_layout_record, write_layer_semantics, write_manifest

COMPONENT_NAME = "CapNInterdigitalTee"
SIMULATION_CONFIG = "coupler-CapNInterdigitalTee-cap_matrix"
CAMPAIGN = "generated_from_cap_matrix"


def build_dataset(
    source_rows: list[dict],
    gds_dir: Path,
    output_dir: Path,
    existing_manifest: Path | None = None,
) -> int:
    """Stage one deterministic GDS per source row, failing on a broken pairing."""
    records = []
    for index, row in enumerate(source_rows):
        source_path = gds_dir / f"capn_{index:04d}.gds"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing generated GDS for source row {index}: {source_path}")
        source_id = f"{CAMPAIGN}/capn_{index:04d}"
        relative_path = Path("raw") / COMPONENT_NAME / CAMPAIGN / source_path.name
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        records.append(
            build_layout_record(
                destination,
                component_name=COMPONENT_NAME,
                gds_path=relative_path.as_posix(),
                source_id=source_id,
                design_options=row["design"]["design_options"],
                simulation_config=SIMULATION_CONFIG,
                campaign=CAMPAIGN,
            )
        )
    new_count = len(records)
    if existing_manifest is not None:
        prior_records = pd.read_parquet(existing_manifest).to_dict(orient="records")
        new_paths = {record["gds_path"] for record in records}
        records = [record for record in prior_records if record["gds_path"] not in new_paths] + records
    manifest_path = output_dir / "metadata" / "manifest.parquet"
    write_manifest(records, manifest_path)
    write_layer_semantics(output_dir / "metadata" / "layer-semantics-v1.json")
    build_geometry_features(pd.read_parquet(manifest_path)).to_parquet(
        output_dir / "metadata" / "geometry-features-v1.parquet", index=False
    )
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gds-dir", type=Path, required=True)
    parser.add_argument("--simulation-json", type=Path)
    parser.add_argument("--existing-manifest", type=Path)
    args = parser.parse_args()

    simulation_json = args.simulation_json or Path(
        hf_hub_download(
            repo_id="SQuADDS/SQuADDS_DB",
            repo_type="dataset",
            filename=f"{SIMULATION_CONFIG}.json",
        )
    )
    count = build_dataset(
        json.loads(simulation_json.read_text()),
        args.gds_dir,
        args.output_dir,
        args.existing_manifest,
    )
    print(f"Staged {count} {COMPONENT_NAME} GDS files.")


if __name__ == "__main__":
    main()
