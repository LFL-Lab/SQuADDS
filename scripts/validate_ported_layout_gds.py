#!/usr/bin/env python
"""Validate representative or complete ported TransmonCross/CapN GDS sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from scripts.generate_simulation_layout_gds import SPECS, _enable_qiskit_metal_pandas_compatibility
from squadds.layouts.geometry_v2 import TERMINAL_PAIRS, encode
from squadds.layouts.qmetal_gds import (
    capn_interdigital_tee_port_markers,
    minimum_ground_clearance_um,
    transmon_cross_port_markers,
    validate_ported_gds,
)

VALIDATION_SPECS = {
    "transmon-cross": {"filename": SPECS["transmon-cross"].filename, "prefix": "transmon_cross"},
    "capn-interdigital-tee": {
        "filename": "coupler-CapNInterdigitalTee-cap_matrix.json",
        "prefix": "capn",
    },
}


def _build(kind: str, row: dict[str, Any]):
    _enable_qiskit_metal_pandas_compatibility()
    from qiskit_metal import designs

    design = designs.DesignPlanar()
    options = row["design"]["design_options"]
    if kind == "transmon-cross":
        from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross

        component = TransmonCross(design, "qubit", options=options)
        markers = transmon_cross_port_markers(component, design)
    else:
        from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

        component = CapNInterdigitalTee(design, "cplr", options=options)
        markers = capn_interdigital_tee_port_markers(component)
    return design, component, markers


def _sample_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, min(total, count), dtype=int).tolist()))


def validate_sweep(
    kind: str,
    gds_dir: Path,
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    check_v2: bool = True,
) -> dict[str, Any]:
    """Validate selected source-row/GDS pairs and return a JSON-safe report."""
    spec = VALIDATION_SPECS[kind]
    results = []
    for index in indices:
        if not 0 <= index < len(rows):
            raise IndexError(f"Row index {index} is outside the source dataset (0..{len(rows) - 1}).")
        path = gds_dir / f"{spec['prefix']}_{index:04d}.gds"
        if not path.is_file():
            results.append({"index": index, "path": str(path), "valid": False, "error": "missing GDS"})
            continue
        try:
            design, component, markers = _build(kind, rows[index])
            result = validate_ported_gds(
                design,
                path,
                markers,
                minimum_ground_clearance_um=minimum_ground_clearance_um(component),
            )
            result["index"] = index
            if check_v2:
                _, metadata = encode(path, rows[index]["design"]["design_options"], return_metadata=True)
                result["v2_terminal_count"] = int(metadata["terminal_count"])
                result["checks"]["v2_two_terminal_discovery"] = metadata["terminal_count"] == 2
                ground_offset = len(TERMINAL_PAIRS)
                ground_availability = metadata["coupling_availability"][ground_offset : ground_offset + 2]
                result["v2_terminal_ground_availability"] = ground_availability
                result["checks"]["v2_two_terminal_ground_spectra"] = ground_availability == [True, True]
                result["valid"] = result["valid"] and all(
                    (
                        result["checks"]["v2_two_terminal_discovery"],
                        result["checks"]["v2_two_terminal_ground_spectra"],
                    )
                )
            results.append(result)
        except Exception as exc:  # report every selected row instead of stopping at the first
            results.append(
                {"index": index, "path": str(path), "valid": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    failures = [result for result in results if not result["valid"]]
    return {
        "kind": kind,
        "source_rows": len(rows),
        "validated_rows": len(results),
        "indices": indices,
        "valid": not failures,
        "failure_count": len(failures),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=sorted(VALIDATION_SPECS))
    parser.add_argument("gds_dir", type=Path)
    parser.add_argument("--source-json", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Validate every source row/GDS pair.")
    selection.add_argument("--indices", help="Comma-separated zero-based row indices.")
    parser.add_argument("--sample-count", type=int, default=9)
    parser.add_argument("--skip-v2", action="store_true", help="Skip the slower v2 terminal-discovery check.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be at least 1")

    spec = VALIDATION_SPECS[args.kind]
    source_json = args.source_json or Path(
        hf_hub_download("SQuADDS/SQuADDS_DB", spec["filename"], repo_type="dataset")
    )
    rows = json.loads(source_json.read_text())
    if args.all:
        indices = list(range(len(rows)))
    elif args.indices:
        indices = sorted(set(int(value.strip()) for value in args.indices.split(",") if value.strip()))
    else:
        indices = _sample_indices(len(rows), args.sample_count)
    report = validate_sweep(args.kind, args.gds_dir, rows, indices, check_v2=not args.skip_v2)
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
