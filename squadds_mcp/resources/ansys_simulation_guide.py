"""
Ansys-backed simulation pathways (Q3D, eigenmode HFSS, sweeps, driven-modal).

These resources mirror ``squadds_mcp/simulation_playbook.py`` JSON for agents that prefer prose.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from squadds_mcp.simulation_playbook import build_simulation_playbook_summary

ANSYS_SIMULATION_OVERVIEW_MARKDOWN = """# SQuADDS Legacy Ansys + Driven-Modal Map

Companion JSON: **`squadds://simulation-playbook-summary`** (compact) • Full nested JSON (includes driven-modal tree): MCP tool **`get_squadds_simulation_playbook`** with **`playbook_variant`** = **`full`**.

Agents **cannot** start Q3D/HFSS remotely through MCP—these URIs encode what to run locally with Metal + AEDT.

---

## 1. Class-level entry (`AnsysSimulator`)

- Construct with an **`Analyzer`** (already narrowed to datasets) plus a device row / options dict normalized by **`AnsysSimulator._normalize_device_dict`** (`design_options*` JSON shapes, merged cavities, setup aliases).
- **`simulate`** prints a deterministic plan (`_run_simulation`):
  - **Coupled** (`analyzer.selected_system` list containing qubit + cavity claw): executes **`simulate_whole_device`** — HFSS eigenmode on cavity geometry plus **Q3D** LOM loops on qubit (and optionally NCap coupler for half-wave geometries when `setup_coupler` is supplied).
  - **Single-component** rows: **`simulate_single_design`** branches between eigenmode-for-cavity vs pure capacitance sweeps (`run_xmon_LOM`, `run_capn_LOM`).
- Defaults for solver knobs live under **`default_lom_options`** (Metal setup mapping to capacitance / Q3D) vs **`default_eigenmode_options`** (HFSS modal setup vocabulary).
- **`sweep`** forwards into **`squadds.simulations.objects.run_sweep`**, multiplexing filenames for CLT vs NCap vs generic parameter grids.
- **`run_drivenmodal`** is the *separate typed-request* subsystem (checkpoint manifest + Metal HFSS driven-modal render path) documented under **`squadds://drivenmodal-workflow`**.

---

## 2. Q3D capacitance (Metal `LOManalysis(..., "q3d")`)

- **`run_xmon_LOM`** instantiates **`TransmonCross`**, configures **`LOManalysis(design, "q3d")`**, meshes with open termination pads gleaned from `connection_pads`, then wraps matrices through **`build_xmon_lom_payload`**.
- **`run_capn_LOM`** meshes **`CapNInterdigitalTee`**; payload builders feed **`normalize_simulation_results`** when stitched to eigenmode data.
- Database rows must stay synchronized—call MCP discovery tools (`get_dataset`, `find_closest_designs`) before tweaking geometry literals.

---

## 3. HFSS eigenmode (function **`run_eigenmode`**)

- Parses cavity claw dictionaries (CLT vs NCap) and drives **`EPRanalysis`** scaffolding to converge modes, yielding rough **`cavity_frequency`**, **`Q`**, **`kappa`** payloads via **`build_eigenmode_payload`**.
- Alone for claw-only eigenmode notebooks, or chained before/after qubit Q3D in coupled flows.

---

## 4. Coupled merges (`simulate_whole_device`)

### CoupledLineTee (quarter-wave feel)

1. **`run_eigenmode(..., cross_dict=...)`** shares qubit anchors with CPW meshes.
2. **`run_xmon_LOM`** for qubit Maxwell caps.
3. **`get_sim_results`** merges eigenmode_df + qubit capacitances.

### Half-wave (`coupler_type` ~ `"ncap"`)

1. Eigenmode invocation with **`coupler_type='ncap'`**.
2. **`run_capn_LOM`** on interdigital coupler options pulled from **`device_dict`**.
3. **`run_xmon_LOM`** for qubit symmetry.

Returned dict keeps renderer breadcrumbs (`lom_renderer_options`, `eigenmode_renderer_options`) mirroring Analyzer persistence semantics.

---

## 5. Parameter sweeps (`run_sweep` / **`AnsysSimulator.sweep`**)

Iterates dictionaries of geometries feeding **`simulate_single_design`** (or equivalents). Expect sequential HFSS+Q3d sessions per row—plan cluster time accordingly.

---

## 6. HFSS driven-modal subsystem

Dedicated typed requests (`CapacitanceExtractionRequest`, `CoupledSystemDrivenModalRequest`) + checkpoint manifests + skrf ingestion differ from Classical LOM/epr lineage. Dive into **`squadds://drivenmodal-workflow`**, playbook URI **`squadds://drivenmodal-playbook`**, MCP helper **`get_drivenmodal_playbook_json`**, and Maxwell clarification **`get_maxwell_capacitance_conventions`**.

---

## Suggested MCP traversal

1. Skim **`squadds://ansys-simulation-overview`** (this file).
2. Optionally fetch **`squadds://simulation-playbook-summary`** or call **`get_squadds_simulation_playbook`** with **`playbook_variant`** = **`summary`** for structured search.
3. Call **`playbook_variant`** = **`full`** when you must ship every driven-modal subtree in single JSON."""


def register_ansys_simulation_resources(mcp: FastMCP) -> None:
    """Register umbrella simulation documentation resources."""

    @mcp.resource("squadds://ansys-simulation-overview")
    async def ansys_simulation_overview_md() -> str:
        """Narrative map of Q3D, eigenmode, coupled merges, sweeps, driven-modal."""
        return ANSYS_SIMULATION_OVERVIEW_MARKDOWN

    @mcp.resource("squadds://simulation-playbook-summary")
    async def simulation_playbook_summary_json() -> str:
        """JSON playbook summary mirroring MCP tool variant `summary` (no nested driven-modal tree)."""
        return json.dumps(build_simulation_playbook_summary(), indent=2)
