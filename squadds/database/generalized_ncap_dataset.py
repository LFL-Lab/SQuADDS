"""Normalize generalized interdigital capacitor sweeps for SQuADDS."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

COMPONENT_NAME = "GeneralizedCapNInterdigital"
CONTRIBUTOR = {
    "group": "Levenson-Falk Lab",
    "PI": "Eli Levenson-Falk",
    "institution": "USC",
    "uploader": "Saikat Das",
    "date_created": "2026-07-25",
}
MATRIX_KEYS = (
    "C_G_G",
    "C_G_N",
    "C_G_S",
    "C_N_G",
    "C_N_N",
    "C_N_S",
    "C_S_G",
    "C_S_N",
    "C_S_S",
)


def _pairwise_results(matrix: dict[str, float]) -> dict[str, float | str | dict[str, float]]:
    return {
        "north_to_north": matrix["C_N_N"],
        "north_to_south": abs(matrix["C_N_S"]),
        "north_to_ground": abs(matrix["C_N_G"]),
        "south_to_south": matrix["C_S_S"],
        "south_to_ground": abs(matrix["C_S_G"]),
        "ground_to_ground": matrix["C_G_G"],
        "top_to_top": matrix["C_N_N"],
        "top_to_bottom": abs(matrix["C_N_S"]),
        "top_to_ground": abs(matrix["C_N_G"]),
        "bottom_to_bottom": matrix["C_S_S"],
        "bottom_to_ground": abs(matrix["C_S_G"]),
        "maxwell_matrix": {key: matrix[key] for key in MATRIX_KEYS},
        "units": "fF",
    }


def validate_source_record(record: dict[str, Any]) -> None:
    """Validate fields and physical matrix invariants required by the converter."""
    required = {"cap_instance_name", "cap_options", "ansys_gds", "cap_matrix"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Source record is missing required fields: {sorted(missing)}")

    matrix = record["cap_matrix"].get("matrix_elements_fF", {})
    if set(matrix) != set(MATRIX_KEYS):
        raise ValueError("Source record must contain the complete 3x3 Maxwell capacitance matrix.")

    for left, right in (("C_G_N", "C_N_G"), ("C_G_S", "C_S_G"), ("C_N_S", "C_S_N")):
        if abs(matrix[left] - matrix[right]) > 1e-9:
            raise ValueError(f"Maxwell matrix is not symmetric at {left}/{right}.")

    if any(matrix[key] <= 0 for key in ("C_G_G", "C_N_N", "C_S_S")):
        raise ValueError("Maxwell matrix diagonal entries must be positive.")
    if any(matrix[key] > 0 for key in ("C_G_N", "C_G_S", "C_N_S")):
        raise ValueError("Maxwell matrix off-diagonal entries must be non-positive.")


def convert_source_record(record: dict[str, Any], campaign: str, source_file: str) -> dict[str, Any]:
    """Convert one raw sweep record to the public SQuADDS row contract."""
    validate_source_record(record)
    q3d = record["ansys_gds"]["q3d_signal"]
    matrix = record["cap_matrix"]["matrix_elements_fF"]
    source_id = f"{campaign}/{record['cap_instance_name']}"

    return {
        "sim_options": {
            "sim_type": "cap_matrix",
            "setup": q3d["setup"],
            "renderer_options": {
                "margin_factor_east_west": record["margin_factor_ew"],
                "margin_factor_north_south": record["margin_factor_ns"],
                "gds_north_port_length": record["gds_north_port_length"],
                "gds_north_port_layer": record["gds_north_port_layer"],
                "gds_south_port_length": record["gds_south_port_length"],
                "gds_south_port_layer": record["gds_south_port_layer"],
                "mesh": record["mesh"],
                "gds_import": q3d,
            },
            "layer_stack": record["layer_stack"],
            "simulator": "Ansys Q3D",
        },
        "sim_results": _pairwise_results(matrix),
        "design": {
            "design_options": record["cap_options"],
            "design_tool": "qiskit-metal",
            "coupler_type": COMPONENT_NAME,
            "component_name": COMPONENT_NAME,
            "component_module": "squadds.components",
            "component_class": COMPONENT_NAME,
            "pin_names": ["north_end", "south_end"],
            "pin_aliases": {"top": "north_end", "bottom": "south_end"},
        },
        "notes": {
            "source_id": source_id,
            "source_campaign": campaign,
            "source_instance_name": record["cap_instance_name"],
            "source_file": source_file,
            "attribution": "Simulated and contributed by Saikat Das, Levenson-Falk Lab, USC.",
        },
        "contributor": dict(CONTRIBUTOR),
    }


def iter_source_files(source_root: Path) -> Iterable[Path]:
    """Yield source JSON files in deterministic campaign/name order."""
    return sorted(source_root.glob("*/*.json"), key=lambda path: (path.parent.name, path.name))


def convert_dataset(source_root: Path, output_path: Path) -> int:
    """Convert an export tree to a deterministic Hugging Face JSON artifact."""
    rows = []
    source_ids = set()
    for path in iter_source_files(source_root):
        with path.open(encoding="utf-8") as source:
            record = json.load(source)
        relative_path = path.relative_to(source_root).as_posix()
        row = convert_source_record(record, path.parent.name, relative_path)
        source_id = row["notes"]["source_id"]
        if source_id in source_ids:
            raise ValueError(f"Duplicate source id: {source_id}")
        source_ids.add(source_id)
        rows.append(row)

    if not rows:
        raise ValueError(f"No JSON files found under {source_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(rows, destination, indent=2)
        destination.write("\n")
    return len(rows)
