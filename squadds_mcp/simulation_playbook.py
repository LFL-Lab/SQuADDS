"""
Unified simulation playbook data for MCP (documentation only — no remote solvers).

Covers legacy Q3D LOM capacitance, HFSS eigenmode (EPR), coupled pipelines,
parameter sweeps, and the HFSS driven-modal subsystem.
"""

from __future__ import annotations

from typing import Any

from squadds_mcp.drivenmodal_playbook import build_drivenmodal_playbook


def build_legacy_ansys_simulation_outline() -> dict[str, Any]:
    """Structural map of Classical SQuADDS Ansys paths (Q3D + eigenmode + sweeps)."""

    return {
        "solver_families": {
            "q3d_lom_capacitance": {
                "metal_api": "`qiskit_metal.analyses.quantization.LOManalysis(design, 'q3d')`",
                "ansys_product": "Ansys Q3D Extractor",
                "typical_outputs": "Capacitance matrix / pairwise caps folded into xmon/NCap payloads",
                "entry_symbols": [
                    "squadds.simulations.objects.run_xmon_LOM",
                    "squadds.simulations.objects.run_capn_LOM",
                ],
                "notes": (
                    "Q3D meshes Metal components with open terminations at chosen pads "
                    "(e.g., TransmonCross connection claws). Returned matrices feed "
                    "`build_xmon_lom_payload` / `build_ncap_lom_payload` before "
                    "`normalize_simulation_results` merges with eigenmode outputs."
                ),
            },
            "hfss_eigenmode_epr": {
                "metal_api": "`EPRanalysis`-driven renders (historical notebooks call `run_eigenmode`)",
                "ansys_product": "Ansys HFSS Eigenmode solve",
                "typical_outputs": (
                    "Rough cavity frequency, quality factor scaffolding, κ estimates via helper plumbing "
                    "(`get_freq_Q_kappa`, cavity claw geometry)"
                ),
                "entry_symbols": ["squadds.simulations.objects.run_eigenmode"],
                "payload_builder": "squadds.simulations.result_normalization.build_eigenmode_payload",
                "notes": (
                    "`run_eigenmode` rebuilds claw/CPW/coupler geometry from `geometry_dict`, "
                    "invokes eigenmode convergence, then packages `sim_results` with cavity_frequency, "
                    "Q / kappa semantics used across the HFSS-only tutorial lineage."
                ),
            },
        },
        "orchestration": {
            "primary_class": (
                "`squadds.simulations.ansys_simulator.AnsysSimulator` — constructed with Analyzer + dataset row / "
                "design_options; owns `simulate`, `sweep`, `run_drivenmodal`, defaults for LOM + eigenmode setups."
            ),
            "simulate_whole_device": {
                "module": "squadds.simulations.objects.simulate_whole_device",
                "when": "`analyzer.selected_system` is coupled list `[qubit, cavity_claw]` (see `simulate` internals)",
                "clt_pipeline": (
                    "`run_eigenmode` on cavity bundle + `run_xmon_LOM` on cross qubit "
                    "`get_sim_results` merges eigenmode_df + LOM dataframe."
                ),
                "ncap_pipeline": (
                    "`run_eigenmode` (`coupler_type='ncap'`) + `run_capn_LOM` on interdigital tee "
                    "+ `run_xmon_LOM` on qubit; triplet stitched in `get_sim_results`."
                ),
                "device_dict_keys_highlight": (
                    "`design_options_qubit`, `design_options_cavity_claw`, "
                    "`setup_qubit`, `setup_cavity_claw`, optional `setup_coupler` for half-wave NCap coupling path."
                ),
            },
            "simulate_single_design": {
                "module": "squadds.simulations.objects.simulate_single_design",
                "when": "Standalone `qubit`, `cavity_claw`, or coupler-centric dataset rows",
                "branching_logic": (
                    "If dataframe contains CPW claw keys ⇒ eigenmode on cavity claw dict; attach NCap-specific "
                    "`run_capn_LOM`; else ⇒ `run_xmon_LOM` or `run_capn_LOM` for pure capacitor extraction datasets."
                ),
            },
            "parameter_sweeps": {
                "symbols": ["squadds.simulations.objects.run_sweep", "AnsysSimulator.sweep"],
                "description": (
                    "`run_sweep` iterates simulator parameter dictionaries, invoking `simulate_single_design` "
                    "(or equivalents) — filenames split by CLT vs NCap vs legacy xmon sweeps."
                ),
            },
            "simulate_method_guardrails": (
                "`AnsysSimulator.simulate` surfaces clear `RuntimeError` when payloads are empty; "
                "async invocation remains `NotImplementedError` pending parallel scheduler work."
            ),
        },
        "datasets_linkage": (
            "Legacy rows surface `setup`, `setup_qubit`, `setup_cavity_claw`, capacitance pair labels, eigenmode quantities — "
            "call existing MCP lookup tools (`get_dataset`, `find_closest_designs`) BEFORE editing locally."
        ),
    }


def build_simulation_playbook_summary() -> dict[str, Any]:
    """Lightweight playbook for MCP resources / quick lookups."""

    return {
        "mcp_execution_notice": (
            "MCP tools document SQuADDS simulation architecture only. Launching Ansys AEDT/Q3D/HFSS remains local."
        ),
        "flow_index": [
            {
                "id": "q3d_capacitance_lom",
                "label": "Q3D / LOM capacitance matrices",
                "depth_resource": "`squadds://ansys-simulation-overview` section 2",
                "symbols": ["run_xmon_LOM", "run_capn_LOM", "LOManalysis(..., 'q3d')"],
            },
            {
                "id": "hfss_eigenmode",
                "label": "HFSS eigenmode cavity extraction",
                "depth_resource": "`squadds://ansys-simulation-overview` section 3",
                "symbols": ["run_eigenmode", "simulate_single_design branches with CPW claws"],
            },
            {
                "id": "legacy_coupled",
                "label": "Coupled qubit + cavity simultaneous Q3D + eigenmode",
                "depth_resource": "`squadds://ansys-simulation-overview` section 4",
                "symbols": ["simulate_whole_device", "AnsysSimulator.simulate"],
            },
            {
                "id": "parameter_sweep",
                "label": "Multi-design sweeps via run_sweep / AnsysSimulator.sweep",
                "symbols": ["run_sweep", "AnsysSimulator.sweep"],
            },
            {
                "id": "hfss_driven_modal",
                "label": "Typed HFSS driven-modal + checkpoints + skrf pipelines",
                "depth_resources": ["`squadds://drivenmodal-workflow`", "`squadds://drivenmodal-playbook`"],
                "tool": "Prefer `get_squadds_simulation_playbook(playbook_variant='full')` for nested driven-modal blob "
                "OR `get_drivenmodal_playbook_json` shortcut.",
            },
        ],
        "legacy_ansys": build_legacy_ansys_simulation_outline(),
    }


def build_simulation_playbook_full() -> dict[str, Any]:
    """Full playbook nesting the driven-modal deep tree."""

    base = build_simulation_playbook_summary()
    base["hfss_driven_modal_playbook_nested"] = build_drivenmodal_playbook()
    return base
