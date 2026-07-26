#!/usr/bin/env python
"""Merge rendered simulation layouts into a SQuADDS layout-dataset release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from scripts.generate_simulation_layout_gds import SPECS
from squadds.layouts import build_geometry_features, build_layout_record, write_manifest

RELEASE_SPECS = {
    "cavity-claw": {
        "component": "cavity",
        "component_name": "CavityClawRouteMeander",
        "campaign": "generated_from_cavity_claw_eigenmode",
        "simulation_config": "cavity_claw-RouteMeander-eigenmode",
    },
    "transmon-cross": {
        "component": "qubit",
        "component_name": "TransmonCross",
        "campaign": "generated_from_transmon_cross_cap_matrix",
        "simulation_config": "qubit-TransmonCross-cap_matrix",
    },
}


def stage(
    kind: str,
    gds_dir: Path,
    simulation_json: Path,
    output_root: Path,
    existing_manifest: Path | None = None,
) -> int:
    """Copy a full rendered sweep and refresh manifest plus geometry features."""
    render_spec = SPECS[kind]
    release_spec = RELEASE_SPECS[kind]
    rows = json.loads(simulation_json.read_text())
    records = []
    for index, row in enumerate(rows):
        name = f"{render_spec.prefix}_{index:04d}.gds"
        source = gds_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing rendered GDS for source row {index}: {source}")
        relative = Path("raw") / release_spec["component_name"] / release_spec["campaign"] / name
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            build_layout_record(
                destination,
                component=release_spec["component"],
                component_name=release_spec["component_name"],
                gds_path=relative.as_posix(),
                source_id=f"{release_spec['campaign']}/{name.removesuffix('.gds')}",
                design_options=row["design"]["design_options"],
                simulation_config=release_spec["simulation_config"],
                campaign=release_spec["campaign"],
            )
        )

    if existing_manifest is not None:
        prior_records = pd.read_parquet(existing_manifest).to_dict(orient="records")
        records = [
            record
            for record in prior_records
            if record["gds_path"] not in {new_record["gds_path"] for new_record in records}
        ] + records
    manifest_path = output_root / "metadata" / "manifest.parquet"
    write_manifest(records, manifest_path)
    build_geometry_features(pd.read_parquet(manifest_path)).to_parquet(
        output_root / "metadata" / "geometry-features-v1.parquet", index=False
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(RELEASE_SPECS))
    parser.add_argument("gds_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--simulation-json", type=Path)
    parser.add_argument("--existing-manifest", type=Path)
    args = parser.parse_args()

    render_spec = SPECS[args.kind]
    simulation_json = args.simulation_json or Path(
        hf_hub_download("SQuADDS/SQuADDS_DB", render_spec.filename, repo_type="dataset")
    )
    count = stage(args.kind, args.gds_dir, simulation_json, args.output_root, args.existing_manifest)
    print(f"Staged {count} total layout records.")


if __name__ == "__main__":
    main()
