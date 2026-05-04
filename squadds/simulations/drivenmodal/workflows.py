"""User-facing driven-modal workflow helpers.

These helpers keep tutorial notebooks focused on the SQuADDS workflow:

1. choose a reference design,
2. declare the driven-modal solve,
3. run or reuse solver artifacts, and
4. compare extracted parameters against the SQuADDS reference row.

The lower-level renderer and HFSS compatibility helpers remain available for
advanced users, but most notebooks should start here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scqubits as scq
from scipy import constants

from squadds.core.json_utils import deserialize_json_like

from .models import (
    CapacitanceExtractionRequest,
    CoupledSystemDrivenModalRequest,
    DrivenModalArtifactPolicy,
    DrivenModalLayerStackSpec,
    DrivenModalSetupSpec,
    DrivenModalSweepSpec,
)

QUBIT_CLAW_CAPACITANCE_KEYS = [
    "cross_to_ground",
    "claw_to_ground",
    "cross_to_claw",
    "cross_to_cross",
    "claw_to_claw",
    "ground_to_ground",
]

NCAP_CAPACITANCE_KEYS = [
    "top_to_top",
    "top_to_bottom",
    "top_to_ground",
    "bottom_to_bottom",
    "bottom_to_ground",
    "ground_to_ground",
]

HAMILTONIAN_KEYS = [
    "qubit_frequency_ghz",
    "anharmonicity_mhz",
    "cavity_frequency_ghz",
    "kappa_mhz",
    "g_mhz",
    "chi_mhz",
]


@dataclass(frozen=True)
class DrivenModalRuntimePaths:
    """Conventional runtime folders for tutorial and scripted runs."""

    root: str
    checkpoints: str
    hfss_projects: str
    local_analysis: str | None = None


def default_layer_stack() -> DrivenModalLayerStackSpec:
    """Return the SQuADDS cryogenic HFSS layer-stack convention."""
    return DrivenModalLayerStackSpec(
        preset="squadds_hfss_v1",
        chip_name="main",
        metal_thickness_um=0.2,
        substrate_thickness_um=750.0,
    )


def default_artifact_policy(*, resume_existing: bool = True) -> DrivenModalArtifactPolicy:
    """Return the artifact policy used by the driven-modal tutorials."""
    return DrivenModalArtifactPolicy(
        export_touchstone=True,
        export_y_parameters=True,
        export_capacitance_tables=True,
        checkpoint_after_stage=True,
        resume_existing=resume_existing,
    )


def default_capacitance_setup(*, freq_ghz: float = 5.0) -> DrivenModalSetupSpec:
    """Return a Q3D-comparison driven-modal setup for capacitance extraction."""
    return DrivenModalSetupSpec(
        name="DrivenModalSetup",
        freq_ghz=freq_ghz,
        max_delta_s=0.005,
        max_passes=20,
        min_passes=2,
        min_converged=5,
        pct_refinement=30,
        basis_order=-1,
    )


def default_capacitance_sweep(
    *,
    start_ghz: float = 1.0,
    stop_ghz: float = 10.0,
    count: int = 400,
) -> DrivenModalSweepSpec:
    """Return the broad interpolating sweep used for capacitance checks."""
    return DrivenModalSweepSpec(
        name="DrivenModalSweep",
        start_ghz=start_ghz,
        stop_ghz=stop_ghz,
        count=count,
        sweep_type="Interpolating",
        save_fields=False,
        interpolation_tol=0.005,
        interpolation_max_solutions=max(count, 400),
    )


def default_hamiltonian_setup(*, freq_ghz: float) -> DrivenModalSetupSpec:
    """Return a coupled-system setup centered near the expected cavity mode."""
    return DrivenModalSetupSpec(
        name="DrivenModalSetup",
        freq_ghz=freq_ghz,
        max_delta_s=0.005,
        max_passes=20,
        min_passes=2,
        min_converged=7,
        pct_refinement=30,
        basis_order=-1,
    )


def default_hamiltonian_sweep(
    *,
    name: str,
    start_ghz: float,
    stop_ghz: float,
    count: int,
    sweep_type: str = "Interpolating",
) -> DrivenModalSweepSpec:
    """Return one named sweep for a coupled-system Hamiltonian run."""
    return DrivenModalSweepSpec(
        name=name,
        start_ghz=start_ghz,
        stop_ghz=stop_ghz,
        count=count,
        sweep_type=sweep_type,
        save_fields=False,
        interpolation_tol=0.005 if sweep_type == "Interpolating" else None,
        interpolation_max_solutions=max(count, 400) if sweep_type == "Interpolating" else None,
    )


def _row_value(row: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if isinstance(row, pd.Series):
        return row[key]
    return row[key]


def _optional_row_value(row: Mapping[str, Any] | pd.Series, key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _sim_mapping(row: Mapping[str, Any] | pd.Series) -> Mapping[str, Any]:
    sim_results = _optional_row_value(row, "sim_results")
    if isinstance(sim_results, Mapping):
        return sim_results
    return row


def _sim_float(row: Mapping[str, Any] | pd.Series, *keys: str) -> float:
    values = _sim_mapping(row)
    for key in keys:
        value = values.get(key) if hasattr(values, "get") else None
        if value is not None:
            return float(value)
    raise KeyError(f"Could not find any of these simulation-result keys: {keys}")


def _design_options_from_reference(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    design = _optional_row_value(row, "design")
    if isinstance(design, Mapping) and "design_options" in design:
        return deserialize_json_like(design["design_options"])

    design_options = _optional_row_value(row, "design_options")
    if design_options is not None:
        return deserialize_json_like(design_options)

    raise KeyError("Reference row does not contain design options.")


def build_capacitance_request(
    row: Mapping[str, Any] | pd.Series,
    *,
    system_kind: str,
    run_id: str,
    layer_stack: DrivenModalLayerStackSpec | None = None,
    setup: DrivenModalSetupSpec | None = None,
    sweep: DrivenModalSweepSpec | None = None,
    artifacts: DrivenModalArtifactPolicy | None = None,
) -> CapacitanceExtractionRequest:
    """Build a capacitance request for a qubit-claw or NCap reference row."""
    if system_kind == "qubit_claw":
        port_mapping = {
            "cross": {
                "component": "xmon",
                "pin": "rect_jj",
                "metadata": {"hfss_target": "junction", "draw_inductor": False},
            },
            "claw": {"component": "xmon", "pin": "readout"},
        }
    elif system_kind == "ncap":
        port_mapping = {
            "top": {"component": "cplr", "pin": "prime_start"},
            "bottom": {"component": "cplr", "pin": "second_end"},
        }
    else:
        raise ValueError("system_kind must be 'qubit_claw' or 'ncap'.")

    return CapacitanceExtractionRequest(
        system_kind=system_kind,
        design_payload={
            "design_options": _design_options_from_reference(row),
            "port_mapping": port_mapping,
        },
        layer_stack=layer_stack or default_layer_stack(),
        setup=setup or default_capacitance_setup(),
        sweep=sweep or default_capacitance_sweep(),
        artifacts=artifacts or default_artifact_policy(),
        metadata={"run_id": run_id},
    )


def capacitance_reference_summary(
    row: Mapping[str, Any] | pd.Series,
    *,
    system_kind: str,
) -> dict[str, float]:
    """Extract the SQuADDS Q3D capacitance summary from a reference row."""
    keys = QUBIT_CLAW_CAPACITANCE_KEYS if system_kind == "qubit_claw" else NCAP_CAPACITANCE_KEYS
    return {key: _sim_float(row, key) for key in keys}


def capacitance_comparison_table(
    *,
    drivenmodal_fF: Mapping[str, float],
    q3d_fF: Mapping[str, float],
) -> pd.DataFrame:
    """Compare driven-modal capacitances against the SQuADDS Q3D row."""
    rows = []
    for key, q3d_value in q3d_fF.items():
        dm_value = float(drivenmodal_fF[key])
        error_pct = np.nan if q3d_value == 0 else 100.0 * (dm_value - q3d_value) / q3d_value
        rows.append(
            {
                "quantity": key,
                "drivenmodal_fF": dm_value,
                "q3d_fF": float(q3d_value),
                "percent_error": error_pct,
            }
        )
    return pd.DataFrame(rows)


def maxwell_matrix_interpretation() -> pd.DataFrame:
    """Return the active-node convention used to avoid capacitance double counting."""
    return pd.DataFrame(
        [
            {
                "entry": "C[cross, cross]",
                "meaning": "self term seen by the cross node",
                "do_not_add_again": "includes mutual capacitance to the claw",
            },
            {
                "entry": "C[cross, claw]",
                "meaning": "negative mutual entry in the Maxwell matrix",
                "do_not_add_again": "use abs(...) only when naming pair capacitances",
            },
            {
                "entry": "cross_to_ground + cross_to_claw",
                "meaning": "total shunt capacitance used by the transmon model",
                "do_not_add_again": "do not also add cross_to_cross",
            },
        ]
    )


def coupled_design_payload(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Normalize a SQuADDS qubit-cavity row into the QubitCavity payload shape."""
    raw_design_options = _optional_row_value(row, "design_options")
    if raw_design_options is not None:
        return deserialize_json_like(raw_design_options)

    if _optional_row_value(row, "design_options_qubit") is not None:
        qubit_options = deserialize_json_like(_row_value(row, "design_options_qubit"))
        cavity_split = deserialize_json_like(_row_value(row, "design_options_cavity_claw"))
        coupler_options = cavity_split.get("coupler_options") or cavity_split.get("cplr_opts") or {}
        cpw_options = cavity_split.get("cpw_options") or cavity_split.get("cpw_opts") or {}

        return {
            "qubit_options": qubit_options,
            "cavity_claw_options": {
                "coupler_type": _row_value(row, "coupler_type"),
                "coupler_options": coupler_options,
                "cpw_opts": {"left_options": cpw_options},
            },
        }

    raise KeyError("Could not resolve a QubitCavity-compatible payload from the reference row.")


def regularize_coupled_design_payload(design_options: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize nested design options before driven-modal rendering."""
    normalized = deserialize_json_like(design_options)
    cavity_options = normalized["cavity_claw_options"]
    cpw_key = "cpw_opts" if "cpw_opts" in cavity_options else "cpw_options"
    cpw_options = deserialize_json_like(cavity_options.setdefault(cpw_key, {}))
    cavity_options[cpw_key] = cpw_options
    left_options = deserialize_json_like(cpw_options.setdefault("left_options", {}))
    cpw_options["left_options"] = left_options
    left_options["lead"] = deserialize_json_like(left_options.get("lead", {}))
    left_options["meander"] = deserialize_json_like(left_options.get("meander", {}))
    return normalized


def build_coupled_system_request(
    row: Mapping[str, Any] | pd.Series,
    *,
    resonator_type: str,
    run_id: str,
    setup: DrivenModalSetupSpec,
    sweep: DrivenModalSweepSpec,
    layer_stack: DrivenModalLayerStackSpec | None = None,
    artifacts: DrivenModalArtifactPolicy | None = None,
) -> CoupledSystemDrivenModalRequest:
    """Build the 3-port driven-modal request for a qubit-cavity-feedline system."""
    resonator_type = resonator_type.lower().removesuffix("_wave")
    if resonator_type not in {"quarter", "half"}:
        raise ValueError("resonator_type must be 'quarter' or 'half'.")

    design_options = regularize_coupled_design_payload(coupled_design_payload(row))
    return CoupledSystemDrivenModalRequest(
        resonator_type=f"{resonator_type}_wave",
        design_payload={
            "design_options": design_options,
            "coupler_type": _row_value(row, "coupler_type"),
            "port_mapping": {
                "feedline_input": {"component": "feedline", "pin": "start"},
                "feedline_output": {"component": "feedline", "pin": "end"},
                "jj": {
                    "component": "qubit_cavity_xmon",
                    "pin": "rect_jj",
                    "metadata": {"hfss_target": "junction", "draw_inductor": False},
                },
            },
        },
        layer_stack=layer_stack or default_layer_stack(),
        setup=setup,
        sweep=sweep,
        artifacts=artifacts or default_artifact_policy(),
        metadata={"run_id": run_id},
    )


def build_segmented_coupled_system_requests(
    row: Mapping[str, Any] | pd.Series,
    *,
    resonator_type: str,
    run_id: str,
    reference: Mapping[str, float],
    setup: DrivenModalSetupSpec | None = None,
    sweeps: Mapping[str, DrivenModalSweepSpec] | None = None,
    layer_stack: DrivenModalLayerStackSpec | None = None,
    artifacts: DrivenModalArtifactPolicy | None = None,
) -> dict[str, CoupledSystemDrivenModalRequest]:
    """Build same-geometry requests for qubit, bridge, and resonator sweeps."""
    active_setup = setup or default_hamiltonian_setup(freq_ghz=float(reference["cavity_frequency_ghz"]))
    active_sweeps = dict(sweeps or segmented_hamiltonian_sweeps(reference))
    return {
        band_name: build_coupled_system_request(
            row,
            resonator_type=resonator_type,
            run_id=f"{run_id}-{band_name}",
            setup=active_setup,
            sweep=sweep,
            layer_stack=layer_stack,
            artifacts=artifacts,
        )
        for band_name, sweep in active_sweeps.items()
    }


def bare_lj_h_from_qubit_options(qubit_options: Mapping[str, Any]) -> float:
    """Return the bare JJ inductance in Henry from a SQuADDS qubit option dict."""
    raw = qubit_options.get("aedt_q3d_inductance") or qubit_options.get("Lj") or qubit_options.get("LJ")
    if raw is None:
        raise KeyError("Qubit options must include 'aedt_q3d_inductance', 'Lj', or 'LJ'.")
    value = float(raw)
    return value if value > 1e-9 else value * 1e-9


def transmon_state_inductances(
    *,
    lj_h: float,
    cross_to_ground_fF: float,
    cross_to_claw_fF: float,
    ng: float = 0.0,
    ncut: int = 30,
) -> dict[str, float]:
    """Compute state-dependent JJ inductances with scqubits."""
    c_total_f = (cross_to_ground_fF + cross_to_claw_fF) * 1e-15
    ec_ghz = constants.e**2 / (2 * c_total_f) / constants.h / 1e9
    phi0_over_2pi = constants.hbar / (2 * constants.e)
    ej_ghz = (phi0_over_2pi**2 / lj_h) / constants.h / 1e9

    transmon = scq.Transmon(EJ=ej_ghz, EC=ec_ghz, ng=ng, ncut=ncut)
    cos_phi = transmon.matrixelement_table("cos_phi_operator", evals_count=2)
    exp_cos_g = float(np.real(cos_phi[0, 0]))
    exp_cos_e = float(np.real(cos_phi[1, 1]))
    return {
        "ec_ghz": float(ec_ghz),
        "ej_ghz": float(ej_ghz),
        "f_q_hz": float(transmon.eigenvals(evals_count=2)[1] - transmon.eigenvals(evals_count=2)[0]) * 1e9,
        "alpha_hz": float(transmon.anharmonicity()) * 1e9,
        "lj_bare_h": float(lj_h),
        "lj_ground_h": float(lj_h / exp_cos_g),
        "lj_excited_h": float(lj_h / exp_cos_e),
    }


def coupled_reference_summary(row: Mapping[str, Any] | pd.Series) -> dict[str, float]:
    """Extract the target/reference Hamiltonian and JJ data from a SQuADDS row."""
    design_options = coupled_design_payload(row)
    qubit_options = deserialize_json_like(design_options["qubit_options"])
    lj_h = bare_lj_h_from_qubit_options(qubit_options)
    state_data = transmon_state_inductances(
        lj_h=lj_h,
        cross_to_ground_fF=_sim_float(row, "cross_to_ground"),
        cross_to_claw_fF=_sim_float(row, "cross_to_claw"),
    )
    return {
        "qubit_frequency_ghz": _sim_float(row, "qubit_frequency_GHz", "qubit_frequency_ghz"),
        "anharmonicity_mhz": _sim_float(row, "anharmonicity_MHz", "anharmonicity_mhz"),
        "cavity_frequency_ghz": _sim_float(row, "cavity_frequency_GHz", "cavity_frequency_ghz"),
        "kappa_mhz": _sim_float(row, "kappa_kHz", "kappa_khz") / 1000.0,
        "g_mhz": _sim_float(row, "g_MHz", "g_mhz"),
        **state_data,
    }


def segmented_hamiltonian_sweeps(
    reference: Mapping[str, float],
    *,
    qubit_padding_ghz: float = 0.5,
    resonator_padding_ghz: float = 0.5,
    qubit_count: int = 22_000,
    bridge_count: int = 4_001,
    resonator_count: int = 22_000,
) -> dict[str, DrivenModalSweepSpec]:
    """Build fine/coarse/fine sweeps for one coupled-system driven-modal run."""
    qubit_center = float(reference["qubit_frequency_ghz"])
    resonator_center = float(reference["cavity_frequency_ghz"])
    qubit_stop = qubit_center + qubit_padding_ghz
    resonator_start = max(qubit_stop + 0.05, resonator_center - resonator_padding_ghz)
    return {
        "qubit_band": default_hamiltonian_sweep(
            name="QubitFineSweep",
            start_ghz=max(1.0, qubit_center - qubit_padding_ghz),
            stop_ghz=qubit_stop,
            count=qubit_count,
        ),
        "bridge_band": default_hamiltonian_sweep(
            name="BridgeSweep",
            start_ghz=qubit_stop,
            stop_ghz=resonator_start,
            count=bridge_count,
            sweep_type="Fast",
        ),
        "resonator_band": default_hamiltonian_sweep(
            name="ResonatorFineSweep",
            start_ghz=resonator_start,
            stop_ghz=resonator_center + resonator_padding_ghz,
            count=resonator_count,
        ),
    }


def hamiltonian_comparison_table(
    *,
    drivenmodal: Mapping[str, float],
    squadds: Mapping[str, float],
) -> pd.DataFrame:
    """Compare extracted Hamiltonian parameters against the SQuADDS row."""
    rows = []
    for key in HAMILTONIAN_KEYS:
        dm_value = float(drivenmodal[key]) if key in drivenmodal and drivenmodal[key] is not None else np.nan
        ref_value = float(squadds[key]) if key in squadds and squadds[key] is not None else np.nan
        if np.isnan(ref_value) or ref_value == 0:
            error_pct = np.nan
        else:
            error_pct = 100.0 * (dm_value - ref_value) / ref_value
        rows.append(
            {
                "quantity": key,
                "drivenmodal": dm_value,
                "squadds": ref_value,
                "percent_error": error_pct,
            }
        )
    return pd.DataFrame(rows)
