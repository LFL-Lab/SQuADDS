# %% [markdown]
# # Tutorial 12: HFSS Driven-Modal Qubit-Port Admittance for Qubit Extraction
#
# This tutorial reuses the same quarter/half-wave qubit-cavity-feedline geometry
# and 3-port HFSS driven-modal setup from Tutorial 11, but changes the
# post-processing target:
#
# - instead of extracting cavity metrics from loaded `S21`,
# - we inspect the raw qubit-port admittance `Y33`,
# - combine it with a small-signal JJ `R || L || C` model, and
# - extract the linear qubit mode plus the transmon `f01` and `alpha` through
#   `scqubits`.
#
# The goal is to verify whether the qubit frequency can be recovered directly
# from the JJ-port admittance narrative, rather than only from the existing
# capacitance-matrix workflow.

# %%
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from squadds.core.json_utils import deserialize_json_like
from squadds.simulations.drivenmodal.hfss_data import parameter_dataframe_to_tensor
from squadds.simulations.drivenmodal.qubit_admittance import (
    combine_port_admittance_with_jj,
    extract_qubit_from_port_admittance,
)

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - plain Python fallback for non-notebook execution

    def display(obj):
        print(obj)

    def HTML(value):
        return value


THIS_FILE = Path(__file__).resolve()
TUTORIAL11_PATH = THIS_FILE.with_name("Tutorial-11_DrivenModal_Coupled_System_Postprocessing.py")


def load_tutorial11_module():
    spec = importlib.util.spec_from_file_location("tutorial11_dm", TUTORIAL11_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Tutorial 11 module from {TUTORIAL11_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T11 = load_tutorial11_module()


# %% [markdown]
# ## Runtime knobs
#
# `RUN_TAG` is used only for Tutorial 12's own checkpoints, so it can evolve
# independently of Tutorial 11.

# %%
RESONATOR_TYPE = "quarter"  # change to "half" and rerun for the half-wave flow
REFERENCE_INDEX = 0
RUN_TAG = "v1"
FORCE_RERUN = False

QUBIT_DISCOVERY_SWEEP_PADDING_GHZ = 1.0
QUBIT_DISCOVERY_SWEEP_COUNT = 12000
QUBIT_DISCOVERY_CENTER_SHIFT_GHZ = 0.75
QUBIT_MAX_DISCOVERY_ATTEMPTS = 4
QUBIT_FINAL_SWEEP_PADDING_GHZ = 0.25
QUBIT_FINAL_SWEEP_COUNT = 22000
QUBIT_MAX_DELTA_S = 0.005
QUBIT_MIN_CONVERGED = 5

JJ_CAPACITANCE_FF = 2.0
JJ_RESISTANCE_OHMS = 50_000.0
JJ_CAP_SWEEP_FF = [0.0, 0.5, 1.0, 2.0, 4.0]
JJ_RES_SWEEP_OHMS = [math.inf, 1e6, 1e5, 5e4, 1e4]

RUNTIME_ROOT = Path("tutorials/runtime/drivenmodal_qubit_admittance")
CHECKPOINT_ROOT = RUNTIME_ROOT / "checkpoints"
HFSS_PROJECT_ROOT = RUNTIME_ROOT / "hfss_projects"


def configure_tutorial11_runtime(module) -> None:
    module.RESONATOR_TYPE = RESONATOR_TYPE
    module.REFERENCE_INDEX = REFERENCE_INDEX
    module.RUN_TAG = RUN_TAG
    module.FORCE_RERUN = FORCE_RERUN
    module.RUNTIME_ROOT = RUNTIME_ROOT
    module.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    module.HFSS_PROJECT_ROOT = HFSS_PROJECT_ROOT


configure_tutorial11_runtime(T11)


# %% [markdown]
# ## Helpers


# %%
def ensure_runtime_dirs() -> None:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    HFSS_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_plotly_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def format_resistance_label(rj_ohms: float) -> str:
    if math.isinf(rj_ohms):
        return "open"
    if rj_ohms >= 1e6:
        return f"{rj_ohms / 1e6:g} Mohm"
    if rj_ohms >= 1e3:
        return f"{rj_ohms / 1e3:g} kohm"
    return f"{rj_ohms:g} ohm"


def build_qubit_setup_and_sweep(center_ghz: float, *, padding_ghz: float, count: int):
    setup = T11.DrivenModalSetupSpec(
        name=T11.SETUP_TEMPLATE.name,
        freq_ghz=center_ghz,
        max_delta_s=QUBIT_MAX_DELTA_S,
        max_passes=T11.SETUP_TEMPLATE.max_passes,
        min_passes=T11.SETUP_TEMPLATE.min_passes,
        min_converged=QUBIT_MIN_CONVERGED,
        pct_refinement=T11.SETUP_TEMPLATE.pct_refinement,
        basis_order=T11.SETUP_TEMPLATE.basis_order,
    )
    sweep = T11.DrivenModalSweepSpec(
        name=T11.SWEEP_TEMPLATE.name,
        start_ghz=max(1.0, center_ghz - padding_ghz),
        stop_ghz=center_ghz + padding_ghz,
        count=count,
        sweep_type=T11.SWEEP_TEMPLATE.sweep_type,
        save_fields=T11.SWEEP_TEMPLATE.save_fields,
        interpolation_tol=T11.SWEEP_TEMPLATE.interpolation_tol,
        interpolation_max_solutions=T11.SWEEP_TEMPLATE.interpolation_max_solutions,
    )
    return setup, sweep


def build_qubit_discovery_setup_and_sweep(center_ghz: float):
    return build_qubit_setup_and_sweep(
        center_ghz,
        padding_ghz=QUBIT_DISCOVERY_SWEEP_PADDING_GHZ,
        count=QUBIT_DISCOVERY_SWEEP_COUNT,
    )


def build_final_qubit_setup_and_sweep(center_ghz: float):
    return build_qubit_setup_and_sweep(
        center_ghz,
        padding_ghz=QUBIT_FINAL_SWEEP_PADDING_GHZ,
        count=QUBIT_FINAL_SWEEP_COUNT,
    )


def build_request(row: pd.Series, *, setup, sweep, run_suffix: str):
    request = T11.build_request(row, setup=setup, sweep=sweep, run_suffix=run_suffix)
    request.metadata["run_id"] = request.metadata["run_id"].replace("tutorial11-", "tutorial12-", 1)
    return request


def load_bare_lj_h(reference_row: pd.Series) -> float:
    qubit_options = T11.normalize_qubit_options(reference_row["design_options_qubit"])
    return T11.get_bare_lj_h(qubit_options)


def build_model_sweep_dataframe(
    freqs_hz: np.ndarray,
    y33_env: np.ndarray,
    *,
    bare_lj_h: float,
    center_hint_hz: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c_ff in JJ_CAP_SWEEP_FF:
        for rj_ohms in JJ_RES_SWEEP_OHMS:
            try:
                extracted = extract_qubit_from_port_admittance(
                    freqs_hz,
                    y33_env,
                    lj_h=bare_lj_h,
                    cj_f=c_ff * 1e-15,
                    rj_ohms=rj_ohms,
                    center_hint_hz=center_hint_hz,
                )
                rows.append(
                    {
                        "jj_capacitance_fF": float(c_ff),
                        "jj_resistance_ohms": float(rj_ohms) if np.isfinite(rj_ohms) else math.inf,
                        "jj_resistance_label": format_resistance_label(rj_ohms),
                        "linear_resonance_ghz": extracted["linear_resonance_hz"] / 1e9,
                        "effective_capacitance_fF": extracted["effective_capacitance_f"] * 1e15,
                        "qubit_frequency_ghz": extracted["qubit_frequency_hz"] / 1e9,
                        "anharmonicity_mhz": extracted["anharmonicity_hz"] / 1e6,
                        "ej_ghz": extracted["ej_ghz"],
                        "ec_ghz": extracted["ec_ghz"],
                        "real_admittance_uS": extracted["real_admittance_s"] * 1e6,
                    }
                )
            except ValueError as exc:
                rows.append(
                    {
                        "jj_capacitance_fF": float(c_ff),
                        "jj_resistance_ohms": float(rj_ohms) if np.isfinite(rj_ohms) else math.inf,
                        "jj_resistance_label": format_resistance_label(rj_ohms),
                        "linear_resonance_ghz": np.nan,
                        "effective_capacitance_fF": np.nan,
                        "qubit_frequency_ghz": np.nan,
                        "anharmonicity_mhz": np.nan,
                        "ej_ghz": np.nan,
                        "ec_ghz": np.nan,
                        "real_admittance_uS": np.nan,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def build_comparison_rows(extracted: dict[str, float], reference: dict[str, float]) -> pd.DataFrame:
    rows = []
    for quantity, reference_value, extracted_value in [
        ("qubit_frequency_ghz", reference["qubit_frequency_ghz"], extracted["qubit_frequency_ghz"]),
        ("anharmonicity_mhz", reference["anharmonicity_mhz"], extracted["anharmonicity_mhz"]),
        ("linear_qubit_mode_ghz", reference["qubit_frequency_ghz"], extracted["linear_resonance_ghz"]),
    ]:
        error = extracted_value - reference_value
        percent_error = np.nan if reference_value == 0 else (error / reference_value) * 100
        rows.append(
            {
                "quantity": quantity,
                "reference": reference_value,
                "drivenmodal": extracted_value,
                "error": error,
                "percent_error": percent_error,
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    *,
    artifacts_dir: Path,
    freqs_hz: np.ndarray,
    y33_env: np.ndarray,
    bare_lj_h: float,
    extracted: dict[str, float],
    model_df: pd.DataFrame,
) -> None:
    freq_ghz = freqs_hz / 1e9
    y_total = combine_port_admittance_with_jj(
        freqs_hz,
        y33_env,
        lj_h=bare_lj_h,
        cj_f=JJ_CAPACITANCE_FF * 1e-15,
        rj_ohms=JJ_RESISTANCE_OHMS,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_ghz, y=np.imag(y33_env), name="Im(Y33) raw"))
    fig.add_trace(
        go.Scatter(
            x=freq_ghz,
            y=np.imag(y_total),
            name=f"Im(Y33 + Y_JJ) with Cj={JJ_CAPACITANCE_FF:g} fF, Rj={format_resistance_label(JJ_RESISTANCE_OHMS)}",
        )
    )
    fig.add_hline(y=0.0, line_dash="dash")
    if not np.isnan(extracted["linear_resonance_ghz"]):
        fig.add_vline(x=extracted["linear_resonance_ghz"], line_dash="dot")
    fig.update_layout(
        title="Qubit-port admittance zero crossing",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Imaginary admittance (S)",
    )
    write_plotly_html(fig, artifacts_dir / "y33_zero_crossing.html")

    valid_df = model_df.dropna(subset=["qubit_frequency_ghz", "anharmonicity_mhz"])
    fig = px.line(
        valid_df,
        x="jj_capacitance_fF",
        y="qubit_frequency_ghz",
        color="jj_resistance_label",
        markers=True,
        title="Extracted qubit frequency vs JJ model",
        labels={
            "jj_capacitance_fF": "JJ capacitance (fF)",
            "qubit_frequency_ghz": "Qubit frequency (GHz)",
            "jj_resistance_label": "JJ resistance",
        },
    )
    write_plotly_html(fig, artifacts_dir / "qubit_frequency_sweep.html")

    fig = px.line(
        valid_df,
        x="jj_capacitance_fF",
        y="anharmonicity_mhz",
        color="jj_resistance_label",
        markers=True,
        title="Extracted anharmonicity vs JJ model",
        labels={
            "jj_capacitance_fF": "JJ capacitance (fF)",
            "anharmonicity_mhz": "Anharmonicity (MHz)",
            "jj_resistance_label": "JJ resistance",
        },
    )
    write_plotly_html(fig, artifacts_dir / "anharmonicity_sweep.html")

    fig = px.line(
        valid_df,
        x="jj_capacitance_fF",
        y="linear_resonance_ghz",
        color="jj_resistance_label",
        markers=True,
        title="Linear JJ-port mode vs JJ model",
        labels={
            "jj_capacitance_fF": "JJ capacitance (fF)",
            "linear_resonance_ghz": "Linear mode (GHz)",
            "jj_resistance_label": "JJ resistance",
        },
    )
    write_plotly_html(fig, artifacts_dir / "linear_mode_sweep.html")


def summarize_selected_model_extraction(
    freqs_hz: np.ndarray,
    y33_env: np.ndarray,
    *,
    bare_lj_h: float,
    center_hint_hz: float,
) -> tuple[dict[str, float], str | None]:
    try:
        extracted = extract_qubit_from_port_admittance(
            freqs_hz,
            y33_env,
            lj_h=bare_lj_h,
            cj_f=JJ_CAPACITANCE_FF * 1e-15,
            rj_ohms=JJ_RESISTANCE_OHMS,
            center_hint_hz=center_hint_hz,
        )
    except ValueError as exc:
        return (
            {
                "linear_resonance_ghz": np.nan,
                "effective_capacitance_fF": np.nan,
                "qubit_frequency_ghz": np.nan,
                "anharmonicity_mhz": np.nan,
                "ej_ghz": np.nan,
                "ec_ghz": np.nan,
                "real_admittance_uS": np.nan,
                "jj_capacitance_fF": JJ_CAPACITANCE_FF,
                "jj_resistance_ohms": JJ_RESISTANCE_OHMS,
                "resonance_at_sweep_edge": False,
            },
            str(exc),
        )

    return (
        {
            "linear_resonance_ghz": extracted["linear_resonance_hz"] / 1e9,
            "effective_capacitance_fF": extracted["effective_capacitance_f"] * 1e15,
            "qubit_frequency_ghz": extracted["qubit_frequency_hz"] / 1e9,
            "anharmonicity_mhz": extracted["anharmonicity_hz"] / 1e6,
            "ej_ghz": extracted["ej_ghz"],
            "ec_ghz": extracted["ec_ghz"],
            "real_admittance_uS": extracted["real_admittance_s"] * 1e6,
            "jj_capacitance_fF": JJ_CAPACITANCE_FF,
            "jj_resistance_ohms": JJ_RESISTANCE_OHMS,
            "resonance_at_sweep_edge": bool(extracted["resonance_at_sweep_edge"]),
        },
        None,
    )


def run_qubit_demo(
    request,
    *,
    reference_summary: dict[str, float],
    bare_lj_h: float,
    center_hint_hz: float,
) -> dict[str, Any]:
    coupled_result = T11.run_coupled_demo(request, reference_summary)
    run_dir = Path(coupled_result["run_dir"])
    artifacts_dir = run_dir / "artifacts"

    y_df = pd.read_pickle(artifacts_dir / "y_parameters.pkl")
    freqs_hz, y_matrices = parameter_dataframe_to_tensor(y_df, matrix_size=3, parameter_prefix="Y")
    y33_env = y_matrices[:, 2, 2]

    extracted_summary, extraction_warning = summarize_selected_model_extraction(
        freqs_hz,
        y33_env,
        bare_lj_h=bare_lj_h,
        center_hint_hz=center_hint_hz,
    )
    model_df = build_model_sweep_dataframe(
        freqs_hz,
        y33_env,
        bare_lj_h=bare_lj_h,
        center_hint_hz=(extracted_summary["linear_resonance_ghz"] * 1e9)
        if not np.isnan(extracted_summary["linear_resonance_ghz"])
        else center_hint_hz,
    )
    model_df.to_csv(artifacts_dir / "qubit_model_sweep.csv", index=False)

    comparison_df = build_comparison_rows(extracted_summary, reference_summary)
    comparison_df.to_csv(artifacts_dir / "qubit_comparison.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "reference": reference_summary,
        "bare_lj_h": bare_lj_h,
        "selected_model": {
            "jj_capacitance_fF": JJ_CAPACITANCE_FF,
            "jj_resistance_ohms": JJ_RESISTANCE_OHMS,
        },
        "selected_model_extracted": extracted_summary,
        "comparison_rows": comparison_df.to_dict(orient="records"),
        "qubit_extraction_warning": extraction_warning,
        "artifacts": {
            "raw_touchstone": coupled_result["artifacts"]["raw_touchstone"],
            "y_pickle": str(artifacts_dir / "y_parameters.pkl"),
            "model_sweep_csv": str(artifacts_dir / "qubit_model_sweep.csv"),
            "comparison_csv": str(artifacts_dir / "qubit_comparison.csv"),
            "y33_zero_crossing_plot": str(artifacts_dir / "y33_zero_crossing.html"),
        },
    }
    dump_json(artifacts_dir / "qubit_admittance_summary.json", summary)
    save_plots(
        artifacts_dir=artifacts_dir,
        freqs_hz=freqs_hz,
        y33_env=y33_env,
        bare_lj_h=bare_lj_h,
        extracted=extracted_summary,
        model_df=model_df,
    )
    return summary


def select_qubit_followup_frequency(summary: dict[str, Any]) -> float | None:
    extracted = summary.get("selected_model_extracted", {})
    frequency_ghz = extracted.get("linear_resonance_ghz")
    if frequency_ghz is None or np.isnan(frequency_ghz):
        return None
    return float(frequency_ghz)


def qubit_discovery_boundary_direction(summary: dict[str, Any], sweep) -> str | None:
    extracted = summary.get("selected_model_extracted", {})
    if not extracted.get("resonance_at_sweep_edge", False):
        return None
    frequency_ghz = extracted.get("linear_resonance_ghz")
    if frequency_ghz is None or np.isnan(frequency_ghz):
        return None
    if abs(frequency_ghz - sweep.start_ghz) <= 1e-6:
        return "lower"
    if abs(frequency_ghz - sweep.stop_ghz) <= 1e-6:
        return "upper"
    return None


# %% [markdown]
# ## Run the HFSS sweep and analyze `Y33`


# %%
def main() -> None:
    ensure_runtime_dirs()
    T11.ensure_runtime_dirs()

    reference_row = T11.load_reference_row(RESONATOR_TYPE, REFERENCE_INDEX)
    reference_summary = T11.build_reference_summary(reference_row)
    bare_lj_h = load_bare_lj_h(reference_row)
    discovery_center_ghz = reference_summary["qubit_frequency_ghz"]

    print("Selected resonator type:", RESONATOR_TYPE)
    print("Coupler type:", reference_row["coupler_type"])
    initial_setup, initial_sweep = build_qubit_discovery_setup_and_sweep(discovery_center_ghz)
    initial_request = build_request(reference_row, setup=initial_setup, sweep=initial_sweep, run_suffix="discovery-01")
    print("Run ID:", initial_request.metadata["run_id"])
    print("Reference design options:")
    print(json.dumps(deserialize_json_like(initial_request.design_payload["design_options"]), indent=2))

    qubit_result = None
    followup_frequency_ghz = None
    center_hint_hz = reference_summary["qubit_frequency_ghz"] * 1e9
    for attempt in range(1, QUBIT_MAX_DISCOVERY_ATTEMPTS + 1):
        discovery_setup, discovery_sweep = build_qubit_discovery_setup_and_sweep(discovery_center_ghz)
        request = build_request(
            reference_row, setup=discovery_setup, sweep=discovery_sweep, run_suffix=f"discovery-{attempt:02d}"
        )
        print(
            f"\nRunning qubit discovery sweep attempt {attempt}/{QUBIT_MAX_DISCOVERY_ATTEMPTS} "
            f"from {discovery_sweep.start_ghz:.6f} to {discovery_sweep.stop_ghz:.6f} GHz..."
        )
        qubit_result = run_qubit_demo(
            request,
            reference_summary=reference_summary,
            bare_lj_h=bare_lj_h,
            center_hint_hz=center_hint_hz,
        )
        comparison_df = pd.DataFrame(qubit_result["comparison_rows"])
        display(comparison_df)

        followup_frequency_ghz = select_qubit_followup_frequency(qubit_result)
        if followup_frequency_ghz is None:
            break

        direction = qubit_discovery_boundary_direction(qubit_result, discovery_sweep)
        if direction is None:
            break

        center_hint_hz = followup_frequency_ghz * 1e9
        if direction == "lower":
            discovery_center_ghz = max(
                1.0 + QUBIT_DISCOVERY_SWEEP_PADDING_GHZ,
                discovery_center_ghz - QUBIT_DISCOVERY_CENTER_SHIFT_GHZ,
            )
        else:
            discovery_center_ghz += QUBIT_DISCOVERY_CENTER_SHIFT_GHZ
        print(
            "Qubit-mode zero crossing remained on the "
            f"{direction} sweep boundary; shifting the discovery center to {discovery_center_ghz:.6f} GHz."
        )

    if followup_frequency_ghz is None:
        final_result = qubit_result
        print("Discovery sweeps did not isolate a qubit-mode zero crossing well enough for a final zoom sweep.")
    else:
        print(f"Discovery sweep located the linear qubit mode near {followup_frequency_ghz:.6f} GHz")
        final_setup, final_sweep = build_final_qubit_setup_and_sweep(followup_frequency_ghz)
        final_request = build_request(reference_row, setup=final_setup, sweep=final_sweep, run_suffix="final")
        print("Running final high-resolution qubit sweep...")
        final_result = run_qubit_demo(
            final_request,
            reference_summary=reference_summary,
            bare_lj_h=bare_lj_h,
            center_hint_hz=followup_frequency_ghz * 1e9,
        )
        comparison_df = pd.DataFrame(final_result["comparison_rows"])
        display(comparison_df)

    run_dir = Path(final_result["run_dir"])
    artifacts_dir = run_dir / "artifacts"
    model_df = pd.read_csv(artifacts_dir / "qubit_model_sweep.csv")
    comparison_df = pd.DataFrame(final_result["comparison_rows"])

    display(comparison_df)
    display(model_df)
    print("Run directory:", run_dir)
    print("Artifacts:")
    for name, value in final_result["artifacts"].items():
        print(f"  - {name}: {value}")
    display(HTML((artifacts_dir / "y33_zero_crossing.html").read_text()))


if __name__ == "__main__":
    main()
