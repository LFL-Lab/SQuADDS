"""Layer-stack presets for driven-modal HFSS workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .models import DrivenModalLayerStackSpec

LAYER_STACK_COLUMNS = [
    "chip_name",
    "layer",
    "datatype",
    "material",
    "thickness",
    "z_coord",
    "fill",
]


def _format_um(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{formatted}um"


def resolve_layer_stack(spec: DrivenModalLayerStackSpec) -> list[dict[str, Any]]:
    """Resolve a fixed material preset into explicit layer-stack rows."""
    if spec.preset != "squadds_hfss_v1":
        raise ValueError(f"Unknown layer-stack preset: {spec.preset}")

    return [
        {
            "chip_name": spec.chip_name,
            "layer": spec.metal_layer,
            "datatype": spec.datatype,
            "material": "pec",
            "thickness": _format_um(spec.metal_thickness_um),
            "z_coord": _format_um(spec.metal_z_coord_um),
            "fill": True,
        },
        {
            "chip_name": spec.chip_name,
            "layer": spec.substrate_layer,
            "datatype": spec.datatype,
            "material": "silicon",
            "thickness": f"-{_format_um(spec.substrate_thickness_um)}",
            "z_coord": _format_um(spec.substrate_z_coord_um),
            "fill": True,
        },
    ]


def build_layer_stack_dataframe(spec: DrivenModalLayerStackSpec) -> pd.DataFrame:
    """Return the resolved layer stack in the column order expected by Qiskit Metal."""
    rows = resolve_layer_stack(spec)
    return pd.DataFrame(rows, columns=LAYER_STACK_COLUMNS)
