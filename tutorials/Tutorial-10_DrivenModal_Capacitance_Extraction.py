# %% [markdown]
# # Tutorial 10: HFSS Driven-Modal Capacitance Extraction for Qubit-Claw and NCap
#
# This tutorial reruns two capacitance-style SQuADDS geometries in **Ansys HFSS
# driven-modal** using **Qiskit Metal** as the geometry/rendering frontend:
#
# 1. a qubit-claw geometry sourced from the `qubit-TransmonCross-cap_matrix`
#    dataset; and
# 2. an NCap geometry sourced from the `coupler-NCap-cap_matrix` dataset.
#
# The goals are:
#
# - make the driven-modal layer stack explicit;
# - checkpoint artifacts so a crash does not force a full restart;
# - save raw S/Y parameter artifacts, Touchstone files, and extracted
#   capacitance-vs-frequency traces; and
# - compare the extracted capacitance matrix against the existing Q3D-backed
#   SQuADDS dataset values.
#
# This file is written as a Python script with `# %%` cells so it can be run in
# VS Code, Spyder, Jupyter, or as a plain Python script on the Windows machine
# that has Ansys HFSS installed.

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from qiskit_metal import Dict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.renderers.renderer_ansys.hfss_renderer import QHFSSRenderer

from squadds.simulations.drivenmodal.artifacts import load_run_manifest, mark_stage_complete
from squadds.simulations.drivenmodal.capacitance import (
    capacitance_dataframe_from_y_sweep,
    capacitance_matrix_from_y,
    maxwell_capacitance_dataframe,
)
from squadds.simulations.drivenmodal.design import (
    connect_renderer_to_new_ansys_design,
    create_multiplanar_design,
    ensure_drivenmodal_setup,
    format_exception_for_console,
    render_drivenmodal_design,
    run_drivenmodal_sweep,
    safe_ansys_design_name,
)
from squadds.simulations.drivenmodal.hfss_data import (
    parameter_dataframe_to_tensor,
    write_touchstone_from_dataframe,
)
from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request
from squadds.simulations.drivenmodal.models import (
    CapacitanceExtractionRequest,
    DrivenModalArtifactPolicy,
    DrivenModalLayerStackSpec,
    DrivenModalSetupSpec,
    DrivenModalSweepSpec,
)
from squadds.simulations.drivenmodal.ports import build_capacitance_port_specs, split_rendered_ports
from squadds.simulations.utils_component_factory import create_ncap_coupler

try:
    from IPython.display import display as display
except ImportError:  # pragma: no cover - plain Python fallback for non-notebook execution

    def display(obj):
        print(obj)


# %% [markdown]
# ## Runtime knobs
#
# `RUN_TAG` is the easiest way to force a brand-new AEDT design while keeping the
# rest of the script unchanged. Leave `FORCE_RERUN = False` for normal resume
# behavior. Set it to `True` only when you intentionally want to regenerate the
# HFSS artifacts for the same `RUN_TAG`.

# %%
RUN_TAG = "v1"
FORCE_RERUN = False

REMOTE_REPO_ID = "SQuADDS/SQuADDS_DB"
QUBIT_CONFIG = "qubit-TransmonCross-cap_matrix"
NCAP_CONFIG = "coupler-NCap-cap_matrix"
REFERENCE_INDEX = 0

LAYER_STACK = DrivenModalLayerStackSpec(
    preset="squadds_hfss_v1",
    chip_name="main",
    metal_thickness_um=0.2,
    substrate_thickness_um=750.0,
)
SETUP = DrivenModalSetupSpec(
    name="DrivenModalSetup",
    freq_ghz=5.0,
    max_delta_s=0.02,
    max_passes=20,
    min_passes=2,
    min_converged=2,
    pct_refinement=30,
    basis_order=1,
)
SWEEP = DrivenModalSweepSpec(
    name="DrivenModalSweep",
    start_ghz=1.0,
    stop_ghz=12.0,
    count=221,
    sweep_type="Fast",
    save_fields=False,
)
ARTIFACTS = DrivenModalArtifactPolicy(
    export_touchstone=True,
    export_y_parameters=True,
    export_capacitance_tables=True,
    checkpoint_after_stage=True,
    resume_existing=True,
)

RUNTIME_ROOT = Path("tutorials/runtime/drivenmodal_capacitance")
CHECKPOINT_ROOT = RUNTIME_ROOT / "checkpoints"
HFSS_PROJECT_ROOT = RUNTIME_ROOT / "hfss_projects"


# %% [markdown]
# ## Helpers


# %%
def ensure_runtime_dirs() -> None:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    HFSS_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)


def stage_is_complete(manifest_path: Path, stage_name: str) -> bool:
    manifest = load_run_manifest(manifest_path)
    return manifest["stages"][stage_name]["status"] == "complete"


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def prepare_renderer_project(renderer: QHFSSRenderer, project_dir: Path, project_name: str) -> Path:
    """Create a fresh HFSS project and save it to an absolute AEDT path.

    This avoids the Windows/pyEPR project-path duplication bug triggered by
    passing ``project_path``/``project_name`` through ``QHFSSRenderer`` options.
    """
    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / f"{project_name}.aedt"
    renderer.start()
    renderer.new_ansys_project()
    renderer.connect_ansys()
    renderer.pinfo.project.save(str(project_file))
    return project_file


def load_reference_row(config_name: str, index: int) -> dict[str, Any]:
    dataset = load_dataset(REMOTE_REPO_ID, config_name, split="train")
    return dataset[index]


def nearest_frequency_index(freqs_hz: np.ndarray, target_ghz: float) -> int:
    return int(np.argmin(np.abs(freqs_hz - target_ghz * 1e9)))


def summarize_qubit_claw(maxwell_df: pd.DataFrame) -> dict[str, float]:
    return {
        "cross_to_ground": abs(maxwell_df.loc["cross", "ground"]) * 1e15,
        "claw_to_ground": abs(maxwell_df.loc["claw", "ground"]) * 1e15,
        "cross_to_claw": abs(maxwell_df.loc["cross", "claw"]) * 1e15,
        "cross_to_cross": abs(maxwell_df.loc["cross", "cross"]) * 1e15,
        "claw_to_claw": abs(maxwell_df.loc["claw", "claw"]) * 1e15,
        "ground_to_ground": abs(maxwell_df.loc["ground", "ground"]) * 1e15,
    }


def summarize_ncap(maxwell_df: pd.DataFrame) -> dict[str, float]:
    return {
        "top_to_top": abs(maxwell_df.loc["top", "top"]) * 1e15,
        "top_to_bottom": abs(maxwell_df.loc["top", "bottom"]) * 1e15,
        "top_to_ground": abs(maxwell_df.loc["top", "ground"]) * 1e15,
        "bottom_to_bottom": abs(maxwell_df.loc["bottom", "bottom"]) * 1e15,
        "bottom_to_ground": abs(maxwell_df.loc["bottom", "ground"]) * 1e15,
        "ground_to_ground": abs(maxwell_df.loc["ground", "ground"]) * 1e15,
    }


def compare_against_reference(extracted: dict[str, float], reference: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, reference_value in reference.items():
        extracted_value = extracted[key]
        error_pct = np.nan
        if reference_value != 0:
            error_pct = 100.0 * (extracted_value - reference_value) / reference_value
        rows.append(
            {
                "quantity": key,
                "drivenmodal_fF": extracted_value,
                "q3d_dataset_fF": reference_value,
                "percent_error": error_pct,
            }
        )
    return pd.DataFrame(rows)


def plot_capacitance_traces(cap_df: pd.DataFrame, title: str, entries: list[str]) -> None:
    freqs_ghz = cap_df["frequency_hz"].to_numpy(dtype=float) / 1e9
    plt.figure(figsize=(10, 4))
    for entry in entries:
        plt.plot(freqs_ghz, cap_df[entry] * 1e15, label=entry.replace("_F", ""))
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Capacitance (fF)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_qubit_claw_request(reference_row: dict[str, Any]) -> CapacitanceExtractionRequest:
    run_id = f"tutorial10-qubit-claw-{REFERENCE_INDEX:03d}-{RUN_TAG}"
    return CapacitanceExtractionRequest(
        system_kind="qubit_claw",
        design_payload={
            "design_options": reference_row["design"]["design_options"],
            "port_mapping": {
                "cross": {
                    "component": "xmon",
                    "pin": "rect_jj",
                    "metadata": {"hfss_target": "junction", "draw_inductor": False},
                },
                "claw": {"component": "xmon", "pin": "readout"},
            },
        },
        layer_stack=LAYER_STACK,
        setup=SETUP,
        sweep=SWEEP,
        artifacts=ARTIFACTS,
        metadata={"run_id": run_id},
    )


def build_ncap_request(reference_row: dict[str, Any]) -> CapacitanceExtractionRequest:
    run_id = f"tutorial10-ncap-{REFERENCE_INDEX:03d}-{RUN_TAG}"
    return CapacitanceExtractionRequest(
        system_kind="ncap",
        design_payload={
            "design_options": reference_row["design"]["design_options"],
            "port_mapping": {
                "top": {"component": "cplr", "pin": "prime_start"},
                "bottom": {"component": "cplr", "pin": "second_end"},
            },
        },
        layer_stack=LAYER_STACK,
        setup=SETUP,
        sweep=SWEEP,
        artifacts=ARTIFACTS,
        metadata={"run_id": run_id},
    )


def build_qubit_claw_design(request: CapacitanceExtractionRequest, layer_stack_csv: Path):
    design, csv_path = create_multiplanar_design(
        layer_stack=request.layer_stack,
        layer_stack_path=layer_stack_csv,
        enable_renderers=True,
    )
    TransmonCross(design, "xmon", options=Dict(request.design_payload["design_options"]))
    design.rebuild()
    return design, csv_path


def build_ncap_design(request: CapacitanceExtractionRequest, layer_stack_csv: Path):
    design, csv_path = create_multiplanar_design(
        layer_stack=request.layer_stack,
        layer_stack_path=layer_stack_csv,
        enable_renderers=True,
    )
    create_ncap_coupler(dict(request.design_payload["design_options"]), design)
    design.rebuild()
    return design, csv_path


def run_capacitance_demo(
    *,
    label: str,
    request: CapacitanceExtractionRequest,
    build_design_fn,
    node_names: list[str],
    summarize_fn,
    reference_summary: dict[str, float],
) -> dict[str, Any]:
    prepared = run_drivenmodal_request(request, checkpoint_dir=CHECKPOINT_ROOT)
    run_dir = Path(prepared["manifest"]["run_dir"])
    manifest_path = run_dir / "manifest.json"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    s_pickle = artifacts_dir / "s_parameters.pkl"
    y_pickle = artifacts_dir / "y_parameters.pkl"
    z_pickle = artifacts_dir / "z_parameters.pkl"
    touchstone_path = artifacts_dir / f"{label}.s2p"
    cap_table_path = artifacts_dir / "capacitance_vs_frequency.parquet"
    summary_path = artifacts_dir / "summary.json"

    if FORCE_RERUN:
        print(f"[{label}] FORCE_RERUN=True. Existing artifacts will be overwritten if HFSS reruns.")

    if stage_is_complete(manifest_path, "postprocessed") and not FORCE_RERUN and summary_path.exists():
        print(f"[{label}] Reusing checkpointed postprocessed outputs from {run_dir}")
        cap_df = pd.read_parquet(cap_table_path)
        summary = json.loads(summary_path.read_text())
        return {"capacitance_vs_frequency": cap_df, "summary": summary, "run_dir": run_dir}

    if (
        stage_is_complete(manifest_path, "artifacts_exported")
        and not FORCE_RERUN
        and s_pickle.exists()
        and y_pickle.exists()
    ):
        print(f"[{label}] Loading checkpointed solver artifacts from {run_dir}")
        s_df = pd.read_pickle(s_pickle)
        y_df = pd.read_pickle(y_pickle)
    else:
        project_dir = HFSS_PROJECT_ROOT / request.metadata["run_id"]
        project_dir.mkdir(parents=True, exist_ok=True)
        design, layer_stack_csv = build_design_fn(request, artifacts_dir / "layer_stack.csv")
        dump_json(artifacts_dir / "resolved_layer_stack.json", {"rows": prepared["layer_stack"]})
        ansys_design_name = safe_ansys_design_name(request.metadata["run_id"])

        renderer = None
        try:
            renderer = QHFSSRenderer(
                design,
                initiate=False,
                options=Dict(
                    design_name=ansys_design_name,
                ),
            )
            project_file = prepare_renderer_project(renderer, project_dir, request.metadata["run_id"])
            connect_renderer_to_new_ansys_design(
                renderer,
                ansys_design_name,
                "drivenmodal",
            )
            renderer.clean_active_design()

            port_specs = build_capacitance_port_specs(request.system_kind, request.design_payload)
            port_list, jj_to_port = split_rendered_ports(port_specs)
            render_drivenmodal_design(
                renderer,
                selection=list(design.components.keys()),
                port_list=port_list or None,
                jj_to_port=jj_to_port or None,
                box_plus_buffer=True,
            )
            mark_stage_complete(manifest_path, "rendered")

            setup = ensure_drivenmodal_setup(renderer, **request.setup.to_renderer_kwargs())
            mark_stage_complete(manifest_path, "setup_created")

            run_drivenmodal_sweep(
                renderer,
                setup,
                setup_name=request.setup.name,
                **request.sweep.to_renderer_kwargs(),
            )
            mark_stage_complete(manifest_path, "sweep_completed")

            s_df, y_df, z_df = renderer.get_all_Pparms_matrices(matrix_size=len(port_specs))
            s_df.to_pickle(s_pickle)
            y_df.to_pickle(y_pickle)
            z_df.to_pickle(z_pickle)
            write_touchstone_from_dataframe(s_df, matrix_size=len(port_specs), output_path=touchstone_path)
            dump_json(
                artifacts_dir / "solver_artifacts.json",
                {
                    "touchstone_path": str(touchstone_path),
                    "s_pickle": str(s_pickle),
                    "y_pickle": str(y_pickle),
                    "z_pickle": str(z_pickle),
                    "layer_stack_csv": str(layer_stack_csv),
                    "project_dir": str(project_dir),
                    "project_file": str(project_file),
                    "ansys_design_name": ansys_design_name,
                },
            )
            mark_stage_complete(manifest_path, "artifacts_exported")
        finally:
            if renderer is not None:
                try:
                    renderer.disconnect_ansys()
                except Exception as exc:  # pragma: no cover - best effort cleanup on the HFSS machine
                    print(
                        f"[{label}] Warning while disconnecting Ansys: "
                        f"{format_exception_for_console(exc)}"
                    )

    freqs_hz, y_matrices = parameter_dataframe_to_tensor(y_df, matrix_size=2, parameter_prefix="Y")
    cap_df = capacitance_dataframe_from_y_sweep(freqs_hz, y_matrices, node_names=node_names)
    cap_df.to_parquet(cap_table_path, index=False)

    ref_index = nearest_frequency_index(freqs_hz, request.setup.freq_ghz)
    maxwell_df = maxwell_capacitance_dataframe(
        capacitance_matrix_from_y(freqs_hz[ref_index], y_matrices[ref_index]),
        node_names=node_names,
    )
    extracted_summary = summarize_fn(maxwell_df)
    comparison_df = compare_against_reference(extracted_summary, reference_summary)
    comparison_df.to_csv(artifacts_dir / "comparison.csv", index=False)

    summary = {
        "reference_frequency_ghz": freqs_hz[ref_index] / 1e9,
        "reference_summary_fF": reference_summary,
        "drivenmodal_summary_fF": extracted_summary,
        "comparison_rows": comparison_df.to_dict(orient="records"),
        "artifacts": {
            "touchstone_path": str(touchstone_path),
            "capacitance_table": str(cap_table_path),
            "comparison_csv": str(artifacts_dir / "comparison.csv"),
        },
    }
    dump_json(summary_path, summary)
    mark_stage_complete(manifest_path, "postprocessed")
    return {"capacitance_vs_frequency": cap_df, "summary": summary, "run_dir": run_dir}


# %% [markdown]
# ## Load reference dataset rows

# %%
ensure_runtime_dirs()

qubit_reference_row = load_reference_row(QUBIT_CONFIG, REFERENCE_INDEX)
ncap_reference_row = load_reference_row(NCAP_CONFIG, REFERENCE_INDEX)

print("Qubit-claw reference design options:")
print(json.dumps(qubit_reference_row["design"]["design_options"], indent=2))
print("\nNCap reference design options:")
print(json.dumps(ncap_reference_row["design"]["design_options"], indent=2))


# %% [markdown]
# ## Run the qubit-claw driven-modal extraction
#
# The qubit-claw run uses:
#
# - one lumped port on the readout claw connector pin; and
# - one lumped JJ port rendered on the transmon junction element.

# %%
qubit_request = build_qubit_claw_request(qubit_reference_row)
qubit_reference_summary = {
    key: float(qubit_reference_row["sim_results"][key])
    for key in [
        "cross_to_ground",
        "claw_to_ground",
        "cross_to_claw",
        "cross_to_cross",
        "claw_to_claw",
        "ground_to_ground",
    ]
}

qubit_result = run_capacitance_demo(
    label="qubit_claw",
    request=qubit_request,
    build_design_fn=build_qubit_claw_design,
    node_names=["cross", "claw"],
    summarize_fn=summarize_qubit_claw,
    reference_summary=qubit_reference_summary,
)

qubit_comparison_df = pd.DataFrame(qubit_result["summary"]["comparison_rows"])
display(qubit_comparison_df)


# %% [markdown]
# ## Run the NCap driven-modal extraction

# %%
ncap_request = build_ncap_request(ncap_reference_row)
ncap_reference_summary = {
    key: float(ncap_reference_row["sim_results"][key])
    for key in [
        "top_to_top",
        "top_to_bottom",
        "top_to_ground",
        "bottom_to_bottom",
        "bottom_to_ground",
        "ground_to_ground",
    ]
}

ncap_result = run_capacitance_demo(
    label="ncap",
    request=ncap_request,
    build_design_fn=build_ncap_design,
    node_names=["top", "bottom"],
    summarize_fn=summarize_ncap,
    reference_summary=ncap_reference_summary,
)

ncap_comparison_df = pd.DataFrame(ncap_result["summary"]["comparison_rows"])
display(ncap_comparison_df)


# %% [markdown]
# ## Plot capacitance-vs-frequency traces

# %%
plot_capacitance_traces(
    qubit_result["capacitance_vs_frequency"],
    title="Qubit-claw capacitance vs frequency",
    entries=[
        "cross__cross_F",
        "cross__claw_F",
        "claw__claw_F",
    ],
)

plot_capacitance_traces(
    ncap_result["capacitance_vs_frequency"],
    title="NCap capacitance vs frequency",
    entries=[
        "top__top_F",
        "top__bottom_F",
        "bottom__bottom_F",
    ],
)


# %% [markdown]
# ## Inspect the explicit layer stack and checkpoint outputs
#
# The simulation artifacts live under:
#
# - `tutorials/runtime/drivenmodal_capacitance/checkpoints/<run_id>/`
# - `tutorials/runtime/drivenmodal_capacitance/hfss_projects/<run_id>/`
#
# Each run directory includes:
#
# - `manifest.json` with resumable stage state;
# - `artifacts/layer_stack.csv` and `artifacts/resolved_layer_stack.json`;
# - `artifacts/*.pkl` for raw complex S/Y/Z matrices;
# - `artifacts/*.s2p` Touchstone exports; and
# - `artifacts/capacitance_vs_frequency.parquet` for fast downstream analysis.

# %%
print("Qubit-claw run directory:", qubit_result["run_dir"])
print("NCap run directory:", ncap_result["run_dir"])

qubit_layer_stack = pd.read_csv(qubit_result["run_dir"] / "artifacts" / "layer_stack.csv")
ncap_layer_stack = pd.read_csv(ncap_result["run_dir"] / "artifacts" / "layer_stack.csv")

print("\nQubit-claw layer stack:")
display(qubit_layer_stack)
print("\nNCap layer stack:")
display(ncap_layer_stack)


# %% [markdown]
# ## Dataset and API outlook
#
# The long-term SQuADDS driven-modal dataset layout for capacitance-style runs is
# expected to include:
#
# - a compact summary row in `SQuADDS_DB` with:
#   - geometry and layer-stack metadata,
#   - solver setup metadata,
#   - reference-frequency capacitance summaries,
#   - links to heavy artifacts, and
#   - provenance for restart/reproduction;
# - heavy sidecar artifacts containing:
#   - Touchstone files,
#   - dense Y-parameter tables,
#   - capacitance-vs-frequency traces, and
#   - checkpoint manifests / postprocessing summaries.
#
# The API direction is:
#
# - `CapacitanceExtractionRequest(...)` to declare geometry, layer stack, setup,
#   sweep, and artifact policy;
# - `AnsysSimulator.run_drivenmodal(request)` or the lower-level driven-modal
#   runner to initialize checkpoint state; and
# - small reusable postprocessing helpers to load S/Y data, compute
#   capacitance-vs-frequency, export Touchstone files, and compare against
#   existing Q3D-backed records.
