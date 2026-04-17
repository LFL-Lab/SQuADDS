"""Driven-modal HFSS request, result, and utility models."""

from .capacitance import capacitance_dataframe_from_y_sweep, capacitance_matrix_from_y
from .coupled_postprocess import (
    calculate_chi_hz,
    calculate_g_from_chi,
    calculate_kappa_hz,
    calculate_loaded_q,
)
from .layer_stack import LAYER_STACK_COLUMNS, build_layer_stack_dataframe, resolve_layer_stack
from .models import (
    CapacitanceExtractionRequest,
    CapacitanceExtractionResult,
    CoupledSystemDrivenModalRequest,
    CoupledSystemDrivenModalResult,
    DrivenModalArtifactPolicy,
    DrivenModalLayerStackSpec,
    DrivenModalPortSpec,
    DrivenModalRunManifest,
    DrivenModalSetupSpec,
    DrivenModalSweepSpec,
)

__all__ = [
    "LAYER_STACK_COLUMNS",
    "CapacitanceExtractionRequest",
    "CapacitanceExtractionResult",
    "calculate_chi_hz",
    "calculate_g_from_chi",
    "calculate_kappa_hz",
    "calculate_loaded_q",
    "capacitance_dataframe_from_y_sweep",
    "capacitance_matrix_from_y",
    "CoupledSystemDrivenModalRequest",
    "CoupledSystemDrivenModalResult",
    "DrivenModalArtifactPolicy",
    "DrivenModalLayerStackSpec",
    "DrivenModalPortSpec",
    "DrivenModalRunManifest",
    "DrivenModalSetupSpec",
    "DrivenModalSweepSpec",
    "build_layer_stack_dataframe",
    "resolve_layer_stack",
]
