#!/usr/bin/env python
"""Generate resumable GDS artifacts for the CapNInterdigitalTee dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from qiskit_metal import designs
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

from scripts.generate_simulation_layout_gds import _enable_qiskit_metal_pandas_compatibility
from squadds.layouts.qmetal_gds import (
    capn_interdigital_tee_port_markers,
    export_qgeometry_gds,
    minimum_ground_clearance_um,
)

DEFAULT_DATASET = "SQuADDS/SQuADDS_DB"
DEFAULT_FILENAME = "coupler-CapNInterdigitalTee-cap_matrix.json"


def export_row(row: dict, output_path: Path) -> None:
    """Build one legacy coupler and write it atomically as a GDS artifact."""
    _enable_qiskit_metal_pandas_compatibility()
    design = designs.DesignPlanar()
    component = CapNInterdigitalTee(design, "cplr", options=row["design"]["design_options"])
    export_qgeometry_gds(
        design,
        output_path,
        markers=capn_interdigital_tee_port_markers(component),
        include_ground_domain=True,
        minimum_ground_clearance_um=minimum_ground_clearance_um(component),
    )


def generate(output_dir: Path, source_json: Path, limit: int | None = None, overwrite: bool = False) -> tuple[int, int]:
    """Generate missing files and return ``(generated, skipped)``."""
    rows = json.loads(source_json.read_text())
    if limit is not None:
        rows = rows[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = skipped = 0
    for index, row in enumerate(rows):
        output_path = output_dir / f"capn_{index:04d}.gds"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        export_row(row, output_path)
        generated += 1
    return generated, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where capn_0000.gds through capn_0893.gds are written.",
    )
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_json = args.source_json or Path(
        hf_hub_download(
            repo_id=DEFAULT_DATASET,
            repo_type="dataset",
            filename=DEFAULT_FILENAME,
        )
    )
    generated, skipped = generate(args.output_dir, source_json, args.limit, args.overwrite)
    print(f"Generated {generated} GDS files; skipped {skipped} existing files.")


if __name__ == "__main__":
    main()
