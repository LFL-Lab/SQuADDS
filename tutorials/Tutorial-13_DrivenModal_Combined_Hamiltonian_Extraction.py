# %% [markdown]
# # Tutorial 13: Unified Driven-Modal Hamiltonian Extraction
#
# This tutorial combines the cavity-postprocessing and qubit-admittance ideas
# into one end-to-end workflow:
#
# 1. query a quarter/half-wave qubit-cavity design from SQuADDS for a target
#    Hamiltonian point,
# 2. render that geometry into HFSS once,
# 3. run a single driven-modal setup with three named sweeps:
#    - a fine qubit band,
#    - a coarse bridge band, and
#    - a fine resonator band,
# 4. extract the qubit frequency and anharmonicity from the JJ-port admittance,
# 5. extract the resonator frequency and linewidth from the loaded `S21` notch,
# 6. recover `chi` and `g`, and
# 7. compare the full driven-modal Hamiltonian against the SQuADDS reference.
#
# The point of this tutorial is not just to produce one number. It shows the
# user how the same raw 3-port HFSS dataset can be post-processed in two
# different ways:
#
# - `Y33` around the qubit band for `f_q` and `alpha`
# - loaded `S21` around the cavity band for `f_r`, `kappa`, `chi`, and `g`

# %% [markdown]
# ## Physics idea and software idea
#
# This tutorial is intentionally pedagogical in two directions at once.
#
# **Physics idea.**
# A single 3-port driven-modal solve contains both the qubit-side and
# resonator-side linear response. We do not need separate electromagnetic
# geometries for the qubit and the cavity. Instead, we interrogate different
# parts of the same dataset:
#
# - the JJ-port admittance around the qubit band, and
# - the loaded feedline transmission around the cavity band.
#
# **Software idea.**
# The reusable SQuADDS driven-modal API should make that workflow explicit:
# one declared design, one declared layer stack, one setup, multiple named
# sweeps, and post-processing helpers that can be re-run locally from the saved
# artifacts without having to re-open HFSS.

# %%
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import HTML, display
except ImportError:  # pragma: no cover - plain Python fallback for non-notebook execution

    def display(obj):
        print(obj)

    def HTML(value):
        return value


THIS_FILE = Path(__file__).resolve()
TUTORIAL11_PATH = THIS_FILE.with_name("Tutorial-11_DrivenModal_Coupled_System_Postprocessing.py")
TUTORIAL12_PATH = THIS_FILE.with_name("Tutorial-12_DrivenModal_Qubit_Port_Admittance.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T11 = load_module(TUTORIAL11_PATH, "tutorial11_dm")
T12 = load_module(TUTORIAL12_PATH, "tutorial12_dm")


# %% [markdown]
# ## Runtime knobs
#
# The qubit and cavity bands are swept densely, while the middle band is only
# there to show how the same HFSS setup spans the full frequency interval
# between them. The single HFSS design therefore produces three response
# datasets that can later be stitched or analyzed independently.

# %% [markdown]
# ## What this tutorial is trying to prove
#
# The standard SQuADDS workflow recovers Hamiltonian parameters from a mix of
# Q3D, eigenmode, and transmon-side modeling. Here we test the stronger claim:
# one driven-modal HFSS geometry, one setup, and three carefully chosen sweeps
# can recover the same qubit-cavity Hamiltonian story in one place.
#
# We therefore keep the rendered system fixed and only change which frequency
# window and which post-processing view we use:
#
# - JJ-port admittance around the qubit mode for `f_q` and `alpha`
# - loaded cavity transmission around the resonator mode for `f_r`, `kappa`,
#   `chi`, and `g`

# %%
RESONATOR_TYPE = "quarter"
REFERENCE_INDEX = 0
RUN_TAG = "v1"
FORCE_RERUN = False
MAX_SOLVE_ATTEMPTS = 3

QUBIT_BAND_PADDING_GHZ = 0.5
QUBIT_BAND_COUNT = 22000
BRIDGE_BAND_COUNT = 4001
RESONATOR_BAND_PADDING_GHZ = 0.5
RESONATOR_BAND_COUNT = 22000

RUNTIME_ROOT = Path("tutorials/runtime/drivenmodal_combined_hamiltonian")
CHECKPOINT_ROOT = RUNTIME_ROOT / "checkpoints"
HFSS_PROJECT_ROOT = RUNTIME_ROOT / "hfss_projects"
LOCAL_ANALYSIS_ROOT = RUNTIME_ROOT / "local_analysis"


def configure_tutorial11_runtime(module) -> None:
    module.RESONATOR_TYPE = RESONATOR_TYPE
    module.REFERENCE_INDEX = REFERENCE_INDEX
    module.RUN_TAG = RUN_TAG
    module.FORCE_RERUN = FORCE_RERUN
    module.RUNTIME_ROOT = RUNTIME_ROOT
    module.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    module.HFSS_PROJECT_ROOT = HFSS_PROJECT_ROOT
    module.LOCAL_ANALYSIS_ROOT = LOCAL_ANALYSIS_ROOT


configure_tutorial11_runtime(T11)


# %% [markdown]
# ## Helpers

# %% [markdown]
# ### Query a target design from SQuADDS
#
# The tutorial starts from the same kind of reference row a user would query in
# practice: pick a quarter-wave or half-wave family, choose a database entry
# whose Hamiltonian is close to the target, and use that exact geometry as the
# driven-modal validation case.


# %%
def ensure_runtime_dirs() -> None:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    HFSS_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_combined_setup(reference_summary: dict[str, float]):
    return T11.DrivenModalSetupSpec(
        name=T11.SETUP_TEMPLATE.name,
        freq_ghz=reference_summary["cavity_frequency_ghz"],
        max_delta_s=T11.SETUP_TEMPLATE.max_delta_s,
        max_passes=T11.SETUP_TEMPLATE.max_passes,
        min_passes=T11.SETUP_TEMPLATE.min_passes,
        min_converged=T11.SETUP_TEMPLATE.min_converged,
        pct_refinement=T11.SETUP_TEMPLATE.pct_refinement,
        basis_order=T11.SETUP_TEMPLATE.basis_order,
    )


def build_segmented_sweeps(reference_summary: dict[str, float]) -> dict[str, Any]:
    qubit_center = reference_summary["qubit_frequency_ghz"]
    resonator_center = reference_summary["cavity_frequency_ghz"]
    qubit_stop = qubit_center + QUBIT_BAND_PADDING_GHZ
    resonator_start = max(qubit_stop + 0.05, resonator_center - RESONATOR_BAND_PADDING_GHZ)

    return {
        "qubit_band": T11.DrivenModalSweepSpec(
            name="QubitFineSweep",
            start_ghz=max(1.0, qubit_center - QUBIT_BAND_PADDING_GHZ),
            stop_ghz=qubit_stop,
            count=QUBIT_BAND_COUNT,
            sweep_type="Interpolating",
            save_fields=False,
            interpolation_tol=T11.SWEEP_TEMPLATE.interpolation_tol,
            interpolation_max_solutions=T11.SWEEP_TEMPLATE.interpolation_max_solutions,
        ),
        "bridge_band": T11.DrivenModalSweepSpec(
            name="BridgeSweep",
            start_ghz=qubit_stop,
            stop_ghz=resonator_start,
            count=BRIDGE_BAND_COUNT,
            sweep_type="Fast",
            save_fields=False,
        ),
        "resonator_band": T11.DrivenModalSweepSpec(
            name="ResonatorFineSweep",
            start_ghz=resonator_start,
            stop_ghz=resonator_center + RESONATOR_BAND_PADDING_GHZ,
            count=RESONATOR_BAND_COUNT,
            sweep_type="Interpolating",
            save_fields=False,
            interpolation_tol=T11.SWEEP_TEMPLATE.interpolation_tol,
            interpolation_max_solutions=T11.SWEEP_TEMPLATE.interpolation_max_solutions,
        ),
    }


# %% [markdown]
# ### Why the sweep is segmented
#
# The qubit and cavity signatures live on very different scales. A single dense
# sweep over the whole band would be expensive and still waste most of its
# points where we do not need them. Instead, we use:
#
# - a fine qubit-band sweep for the admittance zero-crossing and its slope,
# - a coarse bridge sweep through the middle of the spectrum, and
# - a fine resonator-band sweep for notch fitting and qubit-state splitting.
#
# All three sweeps are executed on the same rendered HFSS design, so the
# geometry, materials, boundaries, and seed meshes stay consistent.


def build_request(row: pd.Series, *, setup, sweep, run_suffix: str):
    request = T11.build_request(row, setup=setup, sweep=sweep, run_suffix=run_suffix)
    request.metadata["run_id"] = request.metadata["run_id"].replace("tutorial11-", "tutorial13-", 1)
    return request


def export_current_sweep_artifacts(renderer, *, band_dir: Path) -> dict[str, str]:
    band_dir.mkdir(parents=True, exist_ok=True)
    s_df, y_df, z_df = renderer.get_all_Pparms_matrices(matrix_size=3)
    s_pickle = band_dir / "s_parameters.pkl"
    y_pickle = band_dir / "y_parameters.pkl"
    z_pickle = band_dir / "z_parameters.pkl"
    raw_touchstone = band_dir / "raw_coupled_system.s3p"

    s_df.to_pickle(s_pickle)
    y_df.to_pickle(y_pickle)
    z_df.to_pickle(z_pickle)
    T11.write_touchstone_from_dataframe(s_df, matrix_size=3, output_path=raw_touchstone)
    return {
        "s_pickle": str(s_pickle),
        "y_pickle": str(y_pickle),
        "z_pickle": str(z_pickle),
        "raw_touchstone": str(raw_touchstone),
    }


# %% [markdown]
# ### Qubit extraction from the JJ-port admittance
#
# The qubit-band processing mirrors Tutorial 12. We reduce the 3-port
# environment to the JJ port with explicit feedline terminations, attach a small
# signal `R || L || C` JJ model, and then locate the linear mode from the
# `Im[Y_total] = 0` crossing. `scqubits` then converts that linear environment
# into the dressed `f01` and anharmonicity.


def postprocess_qubit_band(
    *,
    y_pickle_path: Path,
    artifacts_dir: Path,
    reference_summary: dict[str, float],
    bare_lj_h: float,
) -> dict[str, Any]:
    y_df = pd.read_pickle(y_pickle_path)
    freqs_hz, y_matrices = T12.parameter_dataframe_to_tensor(y_df, matrix_size=3, parameter_prefix="Y")
    y33_env = T12.reduce_terminated_port_admittance(
        y_matrices,
        target_port=2,
        terminated_port_impedances={
            0: T12.FEEDLINE_TERMINATION_OHMS,
            1: T12.FEEDLINE_TERMINATION_OHMS,
        },
    )
    center_hint_hz = reference_summary["qubit_frequency_ghz"] * 1e9
    extracted_summary, extraction_warning = T12.summarize_selected_model_extraction(
        freqs_hz,
        y33_env,
        bare_lj_h=bare_lj_h,
        center_hint_hz=center_hint_hz,
    )
    model_df = T12.build_model_sweep_dataframe(
        freqs_hz,
        y33_env,
        bare_lj_h=bare_lj_h,
        center_hint_hz=extracted_summary["linear_resonance_ghz"] * 1e9,
    )
    model_df.to_csv(artifacts_dir / "qubit_model_sweep.csv", index=False)
    agreement_df = T12.build_model_agreement_table(model_df, reference_summary)
    agreement_df.to_csv(artifacts_dir / "qubit_model_agreement.csv", index=False)
    termination_df = T12.build_feedline_termination_sensitivity(
        freqs_hz,
        y_matrices,
        bare_lj_h=bare_lj_h,
        center_hint_hz=extracted_summary["linear_resonance_ghz"] * 1e9,
    )
    termination_df.to_csv(artifacts_dir / "feedline_termination_sensitivity.csv", index=False)
    comparison_df = T12.build_comparison_rows(extracted_summary, reference_summary)
    comparison_df.to_csv(artifacts_dir / "qubit_comparison.csv", index=False)
    T12.save_plots(
        artifacts_dir=artifacts_dir,
        freqs_hz=freqs_hz,
        y33_env=y33_env,
        bare_lj_h=bare_lj_h,
        extracted=extracted_summary,
        model_df=model_df,
    )

    summary = {
        "reference": reference_summary,
        "selected_model": {
            "jj_capacitance_fF": T12.JJ_CAPACITANCE_FF,
            "jj_resistance_ohms": T12.JJ_RESISTANCE_OHMS,
        },
        "selected_model_extracted": extracted_summary,
        "reference_effective_capacitance_fF": T12.reference_alpha_to_capacitance_fF(reference_summary),
        "comparison_rows": comparison_df.to_dict(orient="records"),
        "best_model_rows": agreement_df.head(10).to_dict(orient="records"),
        "feedline_termination_rows": termination_df.to_dict(orient="records"),
        "qubit_extraction_warning": extraction_warning,
        "artifacts": {
            "comparison_csv": str(artifacts_dir / "qubit_comparison.csv"),
            "model_sweep_csv": str(artifacts_dir / "qubit_model_sweep.csv"),
            "model_agreement_csv": str(artifacts_dir / "qubit_model_agreement.csv"),
            "feedline_termination_csv": str(artifacts_dir / "feedline_termination_sensitivity.csv"),
            "y33_zero_crossing_plot": str(artifacts_dir / "y33_zero_crossing.html"),
        },
    }
    dump_json(artifacts_dir / "qubit_admittance_summary.json", summary)
    return summary


# %% [markdown]
# ### Resonator extraction from the loaded cavity response
#
# The resonator-band processing mirrors Tutorial 11. We terminate the JJ port
# with the ground-state and excited-state inductances, convert the resulting
# reduced networks back to loaded two-port `S`-parameters, and fit the cavity
# notch. That gives us:
#
# - `f_r` from the loaded notch center,
# - `kappa` from the linewidth,
# - `chi` from the ground/excited resonance split, and
# - `g` from the standard dispersive relation.


def postprocess_resonator_band(
    *,
    run_dir: Path,
    y_pickle_path: Path,
    raw_touchstone_path: Path,
    artifacts_dir: Path,
    reference_summary: dict[str, float],
) -> dict[str, Any]:
    y_df = pd.read_pickle(y_pickle_path)
    freqs_hz, y_matrices = T11.parameter_dataframe_to_tensor(y_df, matrix_size=3, parameter_prefix="Y")
    ground_load = 1j * 2 * np.pi * freqs_hz * reference_summary["lj_ground_h"]
    excited_load = 1j * 2 * np.pi * freqs_hz * reference_summary["lj_excited_h"]

    y_ground = T11.terminate_port_y(y_matrices, terminated_port=2, load_impedance_ohms=ground_load)
    y_excited = T11.terminate_port_y(y_matrices, terminated_port=2, load_impedance_ohms=excited_load)
    s_ground = T11.y_to_s(y_ground, z0_ohms=50.0)
    s_excited = T11.y_to_s(y_excited, z0_ohms=50.0)

    ground_network = T11.network_from_sweep(freqs_hz, s_ground)
    excited_network = T11.network_from_sweep(freqs_hz, s_excited)
    ground_touchstone = artifacts_dir / "loaded_ground.s2p"
    excited_touchstone = artifacts_dir / "loaded_excited.s2p"
    T11.write_network_touchstone(ground_network, ground_touchstone)
    T11.write_network_touchstone(excited_network, excited_touchstone)

    ground_metrics = T11.extract_notch_metrics(freqs_hz, s_ground[:, 1, 0])
    excited_metrics = T11.extract_notch_metrics(freqs_hz, s_excited[:, 1, 0])
    postprocessing_warning = T11.build_postprocessing_warning(ground_metrics, excited_metrics)
    extracted = {
        "cavity_frequency_ghz": np.nan,
        "kappa_mhz": np.nan,
        "g_mhz": np.nan,
        "qubit_frequency_ghz": reference_summary["f_q_hz"] / 1e9,
        "anharmonicity_mhz": reference_summary["alpha_hz"] / 1e6,
        "chi_mhz": np.nan,
    }
    if postprocessing_warning is None:
        chi_hz = T11.calculate_chi_hz(ground_metrics["f_res_hz"], excited_metrics["f_res_hz"])
        loaded_q = T11.calculate_loaded_q(f_res_hz=ground_metrics["f_res_hz"], fwhm_hz=ground_metrics["fwhm_hz"])
        kappa_hz = T11.calculate_kappa_hz(f_res_hz=ground_metrics["f_res_hz"], loaded_q=loaded_q)
        g_rad_s = T11.calculate_g_from_chi(
            f_r_hz=ground_metrics["f_res_hz"],
            f_q_hz=reference_summary["f_q_hz"],
            chi_hz=chi_hz,
            alpha_hz=reference_summary["alpha_hz"],
        )
        extracted.update(
            {
                "cavity_frequency_ghz": ground_metrics["f_res_hz"] / 1e9,
                "kappa_mhz": kappa_hz / 1e6,
                "g_mhz": g_rad_s / (2 * np.pi * 1e6),
                "chi_mhz": chi_hz / 1e6,
            }
        )

    comparison_df = T11.compare_metrics(extracted, reference_summary)
    comparison_df.to_csv(artifacts_dir / "comparison.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "reference": reference_summary,
        "extracted": extracted,
        "ground_metrics": ground_metrics,
        "excited_metrics": excited_metrics,
        "comparison_rows": comparison_df.to_dict(orient="records"),
        "postprocessing_warning": postprocessing_warning,
        "artifacts": {
            "raw_touchstone": str(raw_touchstone_path),
            "loaded_ground_touchstone": str(ground_touchstone),
            "loaded_excited_touchstone": str(excited_touchstone),
            "comparison_csv": str(artifacts_dir / "comparison.csv"),
        },
    }
    summary["local_analysis"] = T11.generate_local_analysis_artifacts(summary)
    dump_json(artifacts_dir / "summary.json", summary)
    return summary


# %% [markdown]
# ### Final Hamiltonian comparison
#
# Once both post-processing branches are complete, we merge them into one
# user-facing comparison table. This is the final checkpoint for the tutorial:
# the full set of extracted driven-modal Hamiltonian parameters, the SQuADDS
# reference values, and the signed/percentage errors for each quantity.


def build_combined_hamiltonian_table(qubit_summary: dict[str, Any], resonator_summary: dict[str, Any]) -> pd.DataFrame:
    reference = resonator_summary["reference"]
    extracted = {
        "qubit_frequency_ghz": qubit_summary["selected_model_extracted"]["qubit_frequency_ghz"],
        "anharmonicity_mhz": qubit_summary["selected_model_extracted"]["anharmonicity_mhz"],
        "cavity_frequency_ghz": resonator_summary["extracted"]["cavity_frequency_ghz"],
        "kappa_mhz": resonator_summary["extracted"]["kappa_mhz"],
        "chi_mhz": resonator_summary["extracted"]["chi_mhz"],
        "g_mhz": resonator_summary["extracted"]["g_mhz"],
    }
    rows = []
    for quantity, reference_value in [
        ("qubit_frequency_ghz", reference["qubit_frequency_ghz"]),
        ("anharmonicity_mhz", reference["anharmonicity_mhz"]),
        ("cavity_frequency_ghz", reference["cavity_frequency_ghz"]),
        ("kappa_mhz", reference["kappa_mhz"]),
        ("g_mhz", reference["g_mhz"]),
        ("chi_mhz", np.nan),
    ]:
        extracted_value = extracted[quantity]
        error = extracted_value - reference_value if not np.isnan(reference_value) else np.nan
        percent_error = np.nan if np.isnan(reference_value) or reference_value == 0 else (error / reference_value) * 100
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


def run_combined_demo(
    request,
    *,
    band_sweeps: dict[str, Any],
    reference_summary: dict[str, float],
    bare_lj_h: float,
) -> dict[str, Any]:
    prepared = T11.run_drivenmodal_request(request, checkpoint_dir=CHECKPOINT_ROOT)
    run_dir = Path(prepared["manifest"]["run_dir"])
    manifest_path = run_dir / "manifest.json"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts_dir / "summary.json"

    if T11.stage_is_complete(manifest_path, "postprocessed") and not FORCE_RERUN and summary_path.exists():
        return json.loads(summary_path.read_text())

    band_artifacts: dict[str, dict[str, str]] = {}
    if not T11.stage_is_complete(manifest_path, "artifacts_exported") or FORCE_RERUN:
        base_project_dir = HFSS_PROJECT_ROOT / request.metadata["run_id"]
        base_project_dir.mkdir(parents=True, exist_ok=True)
        dump_json(artifacts_dir / "resolved_layer_stack.json", {"rows": prepared["layer_stack"]})
        port_specs = T11.build_coupled_system_port_specs(request.design_payload)
        port_list, jj_to_port = T11.split_rendered_ports(port_specs)
        open_pins: list[tuple[str, str]] = []
        last_error: BaseException | None = None

        for attempt in range(1, MAX_SOLVE_ATTEMPTS + 1):
            attempt_label = f"attempt-{attempt:02d}"
            attempt_id = f"{request.metadata['run_id']}-{attempt_label}"
            attempt_project_dir = base_project_dir / attempt_label
            attempt_project_dir.mkdir(parents=True, exist_ok=True)
            renderer = None
            ansys_design_name = T11.safe_ansys_design_name(attempt_id)
            try:
                design, layer_stack_csv = T11.build_coupled_design(request, artifacts_dir / "layer_stack.csv")
                selection = T11.build_render_selection(design, include_launchers=T11.RENDER_LAUNCHERS)
                chip_bounds = T11.apply_buffered_chip_bounds(
                    design,
                    selection=selection,
                    chip_name=request.layer_stack.chip_name,
                )
                T11.save_render_debug_artifacts(
                    design=design,
                    artifacts_dir=artifacts_dir,
                    title=f"Driven-modal combined {RESONATOR_TYPE}-wave Qiskit Metal layout",
                    selection=selection,
                    port_mapping=request.design_payload["port_mapping"],
                    port_list=port_list,
                    jj_to_port=jj_to_port,
                    open_pins=open_pins,
                    chip_bounds=chip_bounds,
                )
                renderer = T11.QHFSSRenderer(design, initiate=False, options=T11.Dict())
                project_file = T11.prepare_renderer_project(renderer, attempt_project_dir, attempt_id)
                T11.connect_renderer_to_new_ansys_design(renderer, ansys_design_name, "drivenmodal")
                renderer.clean_active_design()
                cryo_material = T11.apply_cryo_silicon_material_properties(renderer)
                dump_json(artifacts_dir / "material_properties.json", cryo_material)
                T11.render_drivenmodal_design(
                    renderer,
                    selection=selection,
                    open_pins=open_pins,
                    port_list=port_list or None,
                    jj_to_port=jj_to_port or None,
                    box_plus_buffer=False,
                )
                assigned_perf_e = T11.ensure_perfect_e_boundary(
                    renderer,
                    getattr(renderer, "assign_perfE", []),
                    boundary_name="PerfE_Metal",
                )
                boundary_snapshot = T11.snapshot_boundary_assignments(renderer)
                T11.save_renderer_project(renderer, project_file)
                dump_json(
                    artifacts_dir / f"hfss_render_{attempt_label}.json",
                    {
                        "attempt_label": attempt_label,
                        "ansys_design_name": ansys_design_name,
                        "project_file": str(project_file),
                        "assign_perfE": assigned_perf_e,
                        "boundaries": boundary_snapshot,
                    },
                )
                try:
                    renderer.save_screenshot(path=str(artifacts_dir / f"hfss_render_{attempt_label}.png"), show=False)
                except Exception as exc:
                    dump_json(
                        artifacts_dir / f"hfss_render_{attempt_label}_screenshot_error.json",
                        {"error": T11.format_exception_for_console(exc)},
                    )
                T11.mark_stage_complete(manifest_path, "rendered")

                if hasattr(renderer, "options"):
                    renderer.options.max_mesh_length_port = "7um"
                mesh_lengths = T11.build_coupled_seed_mesh_lengths(design)
                modeler = renderer.pinfo.design.modeler
                mesh_error = None
                try:
                    T11.mesh_objects(modeler, mesh_lengths)
                except Exception as exc:
                    mesh_error = T11.format_exception_for_console(exc)
                dump_json(artifacts_dir / "mesh_lengths.json", {"targets": mesh_lengths, "error": mesh_error})

                setup = T11.ensure_drivenmodal_setup(renderer, **request.setup.to_renderer_kwargs())
                T11.save_renderer_project(renderer, project_file)
                T11.mark_stage_complete(manifest_path, "setup_created")

                band_artifacts = {}
                for band_name, sweep in band_sweeps.items():
                    T11.run_drivenmodal_sweep(
                        renderer, setup, setup_name=request.setup.name, **sweep.to_renderer_kwargs()
                    )
                    T11.save_renderer_project(renderer, project_file)
                    band_dir = artifacts_dir / band_name
                    band_artifacts[band_name] = export_current_sweep_artifacts(renderer, band_dir=band_dir)
                dump_json(
                    artifacts_dir / "solver_artifacts.json",
                    {
                        "band_artifacts": band_artifacts,
                        "layer_stack_csv": str(layer_stack_csv),
                        "project_dir": str(attempt_project_dir),
                        "project_file": str(project_file),
                        "ansys_design_name": ansys_design_name,
                        "attempt_label": attempt_label,
                    },
                )
                T11.mark_stage_complete(manifest_path, "sweep_completed")
                T11.mark_stage_complete(manifest_path, "artifacts_exported")
                break
            except Exception as exc:
                last_error = exc
                dump_json(
                    artifacts_dir / f"solver_{attempt_label}.json",
                    {
                        "status": "failed",
                        "attempt_label": attempt_label,
                        "error": T11.format_exception_for_console(exc),
                    },
                )
                print(
                    f"[{request.metadata['run_id']}] HFSS solve {attempt}/{MAX_SOLVE_ATTEMPTS} failed: "
                    f"{T11.format_exception_for_console(exc)}"
                )
                if attempt == MAX_SOLVE_ATTEMPTS:
                    raise
                print(f"[{request.metadata['run_id']}] Retrying with a fresh internal HFSS design...")
            finally:
                if renderer is not None:
                    try:
                        renderer.disconnect_ansys()
                    except Exception as exc:
                        print(
                            f"[{request.metadata['run_id']}] Warning while disconnecting Ansys: "
                            f"{T11.format_exception_for_console(exc)}"
                        )
                T11.reset_ansys_desktop_processes()
        else:
            if last_error is not None:
                raise last_error
    else:
        solver_artifacts = json.loads((artifacts_dir / "solver_artifacts.json").read_text())
        band_artifacts = solver_artifacts["band_artifacts"]

    qubit_summary = postprocess_qubit_band(
        y_pickle_path=Path(band_artifacts["qubit_band"]["y_pickle"]),
        artifacts_dir=artifacts_dir / "qubit_band",
        reference_summary=reference_summary,
        bare_lj_h=bare_lj_h,
    )
    resonator_summary = postprocess_resonator_band(
        run_dir=run_dir,
        y_pickle_path=Path(band_artifacts["resonator_band"]["y_pickle"]),
        raw_touchstone_path=Path(band_artifacts["resonator_band"]["raw_touchstone"]),
        artifacts_dir=artifacts_dir / "resonator_band",
        reference_summary=reference_summary,
    )
    combined_df = build_combined_hamiltonian_table(qubit_summary, resonator_summary)
    combined_csv = artifacts_dir / "combined_hamiltonian_comparison.csv"
    combined_df.to_csv(combined_csv, index=False)

    summary = {
        "run_dir": str(run_dir),
        "reference": reference_summary,
        "band_artifacts": band_artifacts,
        "qubit_summary": qubit_summary,
        "resonator_summary": resonator_summary,
        "combined_comparison_rows": combined_df.to_dict(orient="records"),
        "artifacts": {
            "combined_comparison_csv": str(combined_csv),
            "qubit_summary_json": str(artifacts_dir / "qubit_band" / "qubit_admittance_summary.json"),
            "resonator_summary_json": str(artifacts_dir / "resonator_band" / "summary.json"),
        },
    }
    dump_json(summary_path, summary)
    T11.mark_stage_complete(manifest_path, "postprocessed")
    return summary


# %% [markdown]
# ## Run the unified HFSS flow and extract the Hamiltonian
#
# We reuse the same raw 3-port simulation data in two ways:
#
# - the qubit-band sweep is reduced to the JJ-port environment admittance and
#   fed through the Tutorial 12 `Y33` + JJ-model extraction, and
# - the resonator-band sweep is terminated with the ground/excited junction
#   inductances and fed through the Tutorial 11 notch extraction.
#
# The final dataframe below is the one-stop summary a user would care about:
# the driven-modal Hamiltonian parameters next to the SQuADDS reference row and
# their error percentages.


# %%
def main() -> None:
    ensure_runtime_dirs()
    T11.ensure_runtime_dirs()

    reference_row = T11.load_reference_row(RESONATOR_TYPE, REFERENCE_INDEX)
    reference_summary = T11.build_reference_summary(reference_row)
    bare_lj_h = T12.load_bare_lj_h(reference_row)
    setup = build_combined_setup(reference_summary)
    band_sweeps = build_segmented_sweeps(reference_summary)
    request = build_request(reference_row, setup=setup, sweep=band_sweeps["qubit_band"], run_suffix="combined")

    print("Selected resonator type:", RESONATOR_TYPE)
    print("Run ID:", request.metadata["run_id"])
    print("Reference design options:")
    print(json.dumps(T11.deserialize_json_like(request.design_payload["design_options"]), indent=2))

    combined_result = run_combined_demo(
        request,
        band_sweeps=band_sweeps,
        reference_summary=reference_summary,
        bare_lj_h=bare_lj_h,
    )

    run_dir = Path(combined_result["run_dir"])
    artifacts_dir = run_dir / "artifacts"
    combined_df = pd.DataFrame(combined_result["combined_comparison_rows"])
    qubit_summary = combined_result["qubit_summary"]
    resonator_summary = combined_result["resonator_summary"]

    display(combined_df)
    display(pd.DataFrame([qubit_summary["selected_model_extracted"]]))
    display(pd.DataFrame(resonator_summary["comparison_rows"]))
    display(pd.DataFrame(qubit_summary["best_model_rows"]))
    display(HTML((artifacts_dir / "qubit_band" / "y33_zero_crossing.html").read_text()))
    resonator_local_analysis = Path(resonator_summary["local_analysis"]["artifacts"]["jj_reference_terminations_html"])
    display(HTML(resonator_local_analysis.read_text()))

    print("Run directory:", run_dir)
    print("Artifacts:")
    for name, value in combined_result["artifacts"].items():
        print(f"  - {name}: {value}")


if __name__ == "__main__":
    main()


# %% [markdown]
# ## License
#
# <div style='width: 100%; background-color:#3cb1c2;color:#324344;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>
#     <h3>This code is a part of SQuADDS</h3>
#     <p>Developed by Sadman Ahmed Shanto</p>
#     <p>This tutorial is written by Sadman Ahmed Shanto and OpenAI Codex</p>
#     <p>&copy; Copyright Sadman Ahmed Shanto & Eli Levenson-Falk 2024.</p>
#     <p>This code is licensed under the MIT License. You may<br> obtain a copy of this license in the LICENSE.txt file in the root directory<br> of this source tree.</p>
#     <p>Any modifications or derivative works of this code must retain this<br>copyright notice, and modified files need to carry a notice indicating<br>that they have been altered from the originals.</p>
# </div>
