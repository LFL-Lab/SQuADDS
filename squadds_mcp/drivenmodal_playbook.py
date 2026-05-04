"""
Structured playbook for HFSS driven-modal workflows (agent-facing metadata).

Separated from MCP wiring so imports stay explicit and JSON shape is tested
without touching the MCP runtime.
"""

from __future__ import annotations

from typing import Any


def build_drivenmodal_playbook() -> dict[str, Any]:
    """Return a JSON-ready map of workflows, APIs, and interpretation notes."""

    return {
        "squadds_package": {
            "root": "squadds.simulations.drivenmodal",
            "tutorial_notebooks_repo_paths": [
                "tutorials/Tutorial-10_DrivenModal_Capacitance_Extraction.ipynb",
                "tutorials/Tutorial-11_DrivenModal_Combined_Hamiltonian_Extraction.ipynb",
                "tutorials/Tutorial-11_DrivenModal_Coupled_System_Postprocessing.py",
                "tutorials/Tutorial-12_DrivenModal_Qubit_Port_Admittance.py",
                "paired .py companions under tutorials/: Tutorial-10_*_Extraction.py, Tutorial-13_*_Extraction.py",
            ],
            "prerequisites": (
                "Local Ansys HFSS + qiskit-metal + pyEPR stack; HDF5/pandas/skrf/scipy/scqubits "
                "for post-processing. Full solving is interactive and not runnable through MCP tools."
            ),
        },
        "workflow_phases": [
            {
                "phase": "select_reference_design",
                "description": (
                    "Start from an Analyzer / SQuADDS_DB dataframe row whose design_options and "
                    "sim_results match the geometry family under test."
                ),
                "mcp_cross_reference": (
                    "Use existing MCP analysis tools (`find_closest_designs`, `get_dataset`) to obtain rows."
                ),
            },
            {
                "phase": "declare_typed_request",
                "description": (
                    "Build a frozen dataclass request: `CapacitanceExtractionRequest` (two-node "
                    "qubit-claw or NCap) or `CoupledSystemDrivenModalRequest` "
                    "(quarter-wave or half-wave qubit+cavity+feedline)."
                ),
                "key_functions": [
                    "squadds.simulations.drivenmodal.workflows.build_capacitance_request",
                    "squadds.simulations.drivenmodal.workflows.build_coupled_system_request",
                    "squadds.simulations.drivenmodal.workflows.build_segmented_coupled_system_requests",
                ],
            },
            {
                "phase": "initialize_checkpointed_run",
                "description": (
                    "`AnsysSimulator.run_drivenmodal(request, checkpoint_dir=..., export_artifacts=...)` "
                    "delegates to `run_drivenmodal_request`: writes manifestJSON, attaches resolved layer-stack CSV "
                    "(preset `squadds_hfss_v1`), prepares resume-friendly artifact layout."
                ),
                "key_functions": [
                    "squadds.simulations.ansys_simulator.AnsysSimulator.run_drivenmodal",
                    "squadds.simulations.drivenmodal.hfss_runner.run_drivenmodal_request",
                ],
            },
            {
                "phase": "render_solve_export",
                "description": (
                    "Tutorial flow uses `render_drivenmodal_design`, "
                    "`ensure_drivenmodal_setup`, `run_drivenmodal_sweep`; exports may include Touchstone "
                    "and wide Y-parameter tables per `DrivenModalArtifactPolicy`."
                ),
                "design_helpers": [
                    "squadds.simulations.drivenmodal.design.render_drivenmodal_design",
                    "squadds.simulations.drivenmodal.design.ensure_drivenmodal_setup",
                    "squadds.simulations.drivenmodal.design.run_drivenmodal_sweep",
                ],
            },
            {
                "phase": "extract_and_compare",
                "description": (
                    "Map HFSS solution tables → numpy tensors / `skrf.Network`, reduce ports with admittance math, "
                    "then compare against SQuADDS Q3D or Hamiltonian-reference columns via comparison tables."
                ),
            },
        ],
        "lumped_ports_reference_impedance": {
            "model": (
                "`DrivenModalPortSpec` (component, pin, impedance_ohms default 50, metadata) emits "
                "Qiskit-Metal tuples via `to_qiskit_port_entry()` → `(component, pin, Z_ref)` "
                "for `port_list` / renderer integration."
            ),
            "junction_port_semantics": (
                "Josephson ports typically use Metal metadata "
                '`{"hfss_target": "junction", "draw_inductor": False}` '
                "(see workflows `build_capacitance_request` / `build_coupled_system_request` port_mapping)."
            ),
            "render_api": (
                "`render_drivenmodal_design(renderer, ..., port_list=..., jj_to_port=..., open_pins=[])` "
                "Normalizes missing `open_pins` when lumped ports are present to satisfy Metal HFSS quirks."
            ),
            "renormalization_note": (
                "HFSS reports S-parameters at each port relative to that port reference impedance. "
                "When converting Y→S uniformly, `squadds.simulations.drivenmodal.coupled_postprocess.y_to_s` "
                "uses one scalar Z0 across all ports — match this to physical feedline impedance for "
                "feedline-facing S11/S21 interpretation. JJ ports use different small-signal physics; "
                "the admittance-centric pipeline (`qubit_admittance`) is the authoritative path for fq/EJ extraction."
            ),
        },
        "capacitance_extraction": {
            "system_kinds": ["qubit_claw", "ncap"],
            "port_mapping_logical_names": {
                "qubit_claw": {"cross": "xmon.rect_jj (junction surrogate)", "claw": "xmon.readout"},
                "ncap": {"top": "cplr.prime_start", "bottom": "cplr.second_end"},
            },
            "pair_cap_keys_qubit_claw": [
                "cross_to_ground",
                "claw_to_ground",
                "cross_to_claw",
                "cross_to_cross",
                "claw_to_claw",
                "ground_to_ground",
            ],
            "pair_cap_keys_ncap": [
                "top_to_top",
                "top_to_bottom",
                "top_to_ground",
                "bottom_to_bottom",
                "bottom_to_ground",
                "ground_to_ground",
            ],
            "defaults": [
                "workflows.default_layer_stack → DrivenModalLayerStackSpec(preset='squadds_hfss_v1', thickness overrides)",
                "workflows.default_capacitance_setup / default_capacitance_sweep (interpolating sweep for broadband Y)",
                "workflows.capacitance_reference_summary(row, system_kind=...) pulls Q3D reference fF floats from dataset row",
                "workflows.capacitance_comparison_table(drivenmodal_fF=..., q3d_fF=...) → percent-error DataFrame",
            ],
            "maxwell_conventions": (
                "Call `maxwell_matrix_interpretation()` (or MCP tool `get_maxwell_capacitance_conventions`) "
                "before narrating capacitor network reduction — avoids double-counting mutual terms vs pair labels."
            ),
        },
        "coupled_system_hamiltonian": {
            "resonator_enum_in_request": ["quarter_wave", "half_wave"],
            "resonator_strings_in_helpers": "`quarter` or `half` in build_coupled_system_request resonator_type",
            "logical_ports_three_port": [
                "feedline_input → feedline.start",
                "feedline_output → feedline.end",
                "jj → qubit_cavity_xmon.rect_jj with junction metadata block",
            ],
            "segmented_sweeps": (
                "`segmented_hamiltonian_sweeps` prepares qubit-band fine, bridge-band fast, cavity-band fine sweeps; "
                "`build_segmented_coupled_system_requests` clones geometry with distinct run_ids per band."
            ),
            "reference_summary": (
                "`coupled_reference_summary(row)` merges SQuADDS Hamiltonian floats with "
                "`transmon_state_inductances` via scqubits.Transmon to expose EC/EJ plus state-dependent LJ."
            ),
            "comparison": "workflows.hamiltonian_comparison_table(drivenmodal=..., squadds=...)",
        },
        "skrf_and_dataframe_pipeline": {
            "parameter_tables": (
                "`parameter_dataframe_to_tensor(frame, matrix_size=n, parameter_prefix)` builds complex tensors from "
                "HFSS-exported columns (`Yij`/`Sij` naming). Rows must index frequency (GHz → Hz scaled internally)."
            ),
            "skrf_network": (
                "`network_from_parameter_dataframe(..., z0_ohms=50)` returns `skrf.Network` "
                "(see `squadds.simulations.drivenmodal.hfss_data`)."
            ),
            "touchstone_export": "write_touchstone_from_dataframe(path, matrix_size=..., z0_ohms=...)",
            "flattened_export_caveat": (
                "Some HDF5/table exports reshuffle tall-vs-wide layouts; tutorials/tests harden parsers against "
                "flattened-row variants — always validate column names vs `parameter_dataframe_to_tensor` expectations."
            ),
        },
        "admittance_reduction_and_loaded_response": {
            "terminate_port_y": (
                "`terminate_port_y(y_matrices, terminated_port=k, load_impedance_ohms=...)`: "
                "reduce admittance dimension after loading one port."
            ),
            "y_to_s": (
                "`y_to_s(y_matrices, z0_ohms=50)` uniform renormalization to S "
                "(see paired note on per-port HFSS Zref vs analytic reduction)."
            ),
            "kappa_loaded_q_helpers": [
                "calculate_loaded_q(f_res_hz, fwhm_hz)",
                "calculate_kappa_hz(f_res_hz, loaded_q)",
                "calculate_chi_hz(f_ground_hz, f_excited_hz)",
                "calculate_g_from_chi(...)",
            ],
            "module": "squadds.simulations.drivenmodal.coupled_postprocess",
        },
        "scqubits_jj_admittance_pipeline": {
            "purpose": (
                "Fit / interpret JJ-port admittance with explicit LJ CJ RJ surrogate and environment Y from HFSS "
                "before mapping to quantized transmon observables."
            ),
            "key_symbols": [
                "jj_parallel_admittance(freqs_hz, lj_h, cj_f, rj_ohms)",
                "combine_port_admittance_with_jj(freqs_hz, y33_env, lj_h=..., cj_f=..., rj_ohms=...)",
                "extract_parallel_mode_from_total_admittance",
                "extract_qubit_from_port_admittance",
            ],
            "transmon_calibration_bridge": (
                "`transmon_state_inductances` (workflows) uses scqubits.Transmon to convert EC/EJ from capacitances + LJ."
            ),
            "module": "squadds.simulations.drivenmodal.qubit_admittance",
        },
        "documentation_uris": {
            "narrative_guide": "squadds://drivenmodal-workflow",
            "machine_playbook_mirror": "squadds://drivenmodal-playbook",
            "umbrella_ansys_overview": "squadds://ansys-simulation-overview",
            "simulation_summary_playbook_uri": "squadds://simulation-playbook-summary",
        },
    }
