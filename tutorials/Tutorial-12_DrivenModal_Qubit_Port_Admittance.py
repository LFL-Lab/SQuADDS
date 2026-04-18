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

QUBIT_SWEEP_PADDING_GHZ = 1.0
QUBIT_SWEEP_COUNT = 22000
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


def build_qubit_setup_and_sweep(reference_summary: dict[str, float]):
    center_ghz = reference_summary["qubit_frequency_ghz"]
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
        start_ghz=max(1.0, center_ghz - QUBIT_SWEEP_PADDING_GHZ),
        stop_ghz=center_ghz + QUBIT_SWEEP_PADDING_GHZ,
        count=QUBIT_SWEEP_COUNT,
        sweep_type=T11.SWEEP_TEMPLATE.sweep_type,
        save_fields=T11.SWEEP_TEMPLATE.save_fields,
        interpolation_tol=T11.SWEEP_TEMPLATE.interpolation_tol,
        interpolation_max_solutions=T11.SWEEP_TEMPLATE.interpolation_max_solutions,
    )
    return setup, sweep


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
    reference_summary: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    center_hint_hz = reference_summary["qubit_frequency_ghz"] * 1e9
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


# %% [markdown]
# ## Run the HFSS sweep and analyze `Y33`


# %%
def main() -> None:
    ensure_runtime_dirs()
    T11.ensure_runtime_dirs()

    reference_row = T11.load_reference_row(RESONATOR_TYPE, REFERENCE_INDEX)
    reference_summary = T11.build_reference_summary(reference_row)
    bare_lj_h = load_bare_lj_h(reference_row)
    setup, sweep = build_qubit_setup_and_sweep(reference_summary)

    print("Selected resonator type:", RESONATOR_TYPE)
    print("Coupler type:", reference_row["coupler_type"])
    request = build_request(reference_row, setup=setup, sweep=sweep, run_suffix="qubit-y33")
    print("Run ID:", request.metadata["run_id"])
    print("Reference design options:")
    print(json.dumps(deserialize_json_like(request.design_payload["design_options"]), indent=2))

    coupled_result = T11.run_coupled_demo(request, reference_summary)
    run_dir = Path(coupled_result["run_dir"])
    artifacts_dir = run_dir / "artifacts"

    y_df = pd.read_pickle(artifacts_dir / "y_parameters.pkl")
    freqs_hz, y_matrices = parameter_dataframe_to_tensor(y_df, matrix_size=3, parameter_prefix="Y")
    y33_env = y_matrices[:, 2, 2]

    extracted = extract_qubit_from_port_admittance(
        freqs_hz,
        y33_env,
        lj_h=bare_lj_h,
        cj_f=JJ_CAPACITANCE_FF * 1e-15,
        rj_ohms=JJ_RESISTANCE_OHMS,
        center_hint_hz=reference_summary["qubit_frequency_ghz"] * 1e9,
    )
    extracted_summary = {
        "linear_resonance_ghz": extracted["linear_resonance_hz"] / 1e9,
        "effective_capacitance_fF": extracted["effective_capacitance_f"] * 1e15,
        "qubit_frequency_ghz": extracted["qubit_frequency_hz"] / 1e9,
        "anharmonicity_mhz": extracted["anharmonicity_hz"] / 1e6,
        "ej_ghz": extracted["ej_ghz"],
        "ec_ghz": extracted["ec_ghz"],
        "real_admittance_uS": extracted["real_admittance_s"] * 1e6,
        "jj_capacitance_fF": JJ_CAPACITANCE_FF,
        "jj_resistance_ohms": JJ_RESISTANCE_OHMS,
    }

    model_df = build_model_sweep_dataframe(
        freqs_hz,
        y33_env,
        bare_lj_h=bare_lj_h,
        reference_summary=reference_summary,
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
        "extracted": extracted_summary,
        "comparison_rows": comparison_df.to_dict(orient="records"),
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

    display(comparison_df)
    display(model_df)
    print("Run directory:", run_dir)
    print("Artifacts:")
    for name, value in summary["artifacts"].items():
        print(f"  - {name}: {value}")
    display(HTML((artifacts_dir / "y33_zero_crossing.html").read_text()))


if __name__ == "__main__":
    main()
