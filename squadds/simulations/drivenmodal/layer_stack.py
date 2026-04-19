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

DEFAULT_CHIP_LAYER_START = "0"
DEFAULT_CHIP_LAYER_END = "2048"
DEFAULT_SAMPLE_HOLDER_TOP_CLEARANCE_UM = 140.0
DEFAULT_SAMPLE_HOLDER_BOTTOM_CLEARANCE_UM = 900.0


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


def resolve_chip_metadata(spec: DrivenModalLayerStackSpec) -> dict[str, str]:
    """Return the chip-level metadata expected by legacy HFSS/Q3D renderers.

    Qiskit Metal's Ansys renderers do not rely on the layer-stack CSV alone. They
    also expect chip material, substrate thickness, and sample-holder extents to
    be present under ``design.chips[chip_name]``. MultiPlanar does not always
    prepopulate those entries, so we derive them from the same preset that
    defines the explicit layer stack.
    """
    if spec.preset != "squadds_hfss_v1":
        raise ValueError(f"Unknown layer-stack preset: {spec.preset}")

    substrate_thickness = _format_um(spec.substrate_thickness_um)
    sample_holder_top = _format_um(spec.substrate_thickness_um + DEFAULT_SAMPLE_HOLDER_TOP_CLEARANCE_UM)
    sample_holder_bottom = _format_um(spec.substrate_thickness_um + DEFAULT_SAMPLE_HOLDER_BOTTOM_CLEARANCE_UM)

    return {
        "material": "silicon",
        "layer_start": DEFAULT_CHIP_LAYER_START,
        "layer_end": DEFAULT_CHIP_LAYER_END,
        "size_z": f"-{substrate_thickness}",
        "sample_holder_top": sample_holder_top,
        "sample_holder_bottom": sample_holder_bottom,
    }
