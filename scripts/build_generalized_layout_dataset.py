#!/usr/bin/env python
"""Stage matched GeneralizedCapNInterdigital GDS files for Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from squadds.layouts import build_layout_record, write_manifest

COMPONENT_NAME = "GeneralizedCapNInterdigital"
SIMULATION_CONFIG = "coupler-GeneralizedCapNInterdigital-cap_matrix"


def _load_rows(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    return {row["notes"]["source_id"]: row for row in rows}


def _parse_source(value: str) -> tuple[str, Path]:
    campaign, separator, raw_path = value.partition("=")
    if not separator or not campaign or not raw_path:
        raise argparse.ArgumentTypeError("GDS sources must use campaign=/absolute/path format.")
    return campaign, Path(raw_path)


def build_dataset(
    source_rows: dict[str, dict], gds_sources: list[tuple[str, Path]], output_dir: Path
) -> tuple[int, int]:
    """Copy matched GDS artifacts and build their manifest; skip unmatched files."""
    records = []
    skipped = 0
    for campaign, source_dir in gds_sources:
        for source_path in sorted(source_dir.glob("*.gds")):
            source_id = f"{campaign}/{source_path.stem}"
            row = source_rows.get(source_id)
            if row is None:
                skipped += 1
                continue
            relative_path = Path("raw") / COMPONENT_NAME / campaign / source_path.name
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
                    campaign=campaign,
                )
            )
    write_manifest(records, output_dir / "metadata" / "manifest.parquet")
    return len(records), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--simulation-json", type=Path)
    parser.add_argument("--gds-source", action="append", required=True, type=_parse_source)
    args = parser.parse_args()

    simulation_json = args.simulation_json or Path(
        hf_hub_download(
            repo_id="SQuADDS/SQuADDS_DB",
            repo_type="dataset",
            filename=f"{SIMULATION_CONFIG}.json",
        )
    )
    matched, skipped = build_dataset(_load_rows(simulation_json), args.gds_source, args.output_dir)
    print(f"Staged {matched} matched GDS files; skipped {skipped} unmatched files.")


if __name__ == "__main__":
    main()
