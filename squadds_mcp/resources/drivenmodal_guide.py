"""
Driven-modal HFSS workflow resources.

Read-only MCP resources that teach agents how ``squadds.simulations.drivenmodal``
matches Qiskit-Metal renders, lumped-port definitions, renormalization, skrf ingestion,
scqubits-based transmon bookkeeping, and SQuADDS dataset comparisons.

Adding a URI
-------------
Prefer extending ``squadds_mcp/drivenmodal_playbook.py`` for structured/agent JSON fields,
then surface significant prose here only when narratives help beyond JSON.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from squadds_mcp.drivenmodal_playbook import build_drivenmodal_playbook

DRIVENMODAL_WORKFLOW_GUIDE_MARKDOWN = """# SQuADDS HFSS Driven-Modal Workflow (Agent Reference)

Full machine-readable outlines also live at `squadds://drivenmodal-playbook` and the MCP tool
`get_drivenmodal_playbook_json`.

---

## Goal

Repeat the published tutorials programmatically:

1. Select a validated SQuADDS database row (`design_options` + `sim_results`).
2. Build a typed **`CapacitanceExtractionRequest`** or **`CoupledSystemDrivenModalRequest`**.
3. Instantiate **`AnsysSimulator`** and call **`run_drivenmodal`** to prepare checkpoint/manifest CSV output.
4. Drive Qiskit-Metal’s HFSS renderer (**`render_drivenmodal_design`**, **`ensure_drivenmodal_setup`**, sweep helpers) locally.
5. Export wide parameter tables / Touchstone, then parse with **`hfss_data.parameter_dataframe_to_tensor`** and **`network_from_parameter_dataframe`** (**scikit-rf / skrf**).
6. Reduce multi-port **`Y`** tensors (**`terminate_port_y`**, **`y_to_s`**) before fitting resonances / line widths.
7. At the JJ, combine HFSS **`Y`** with **`jj_parallel_admittance`** and friends from **`qubit_admittance`**, optionally **`scqubits` Transmon** for EC/EJ or state-dependent JJ inductances via **`transmon_state_inductances`**.
8. Validate against **`capacitance_comparison_table`** or **`hamiltonian_comparison_table`**.

Agents cannot launch HFSS from MCP; resources explain *what correct client code looks like.*

---

## Lumped Ports, `port_list`, and `jj_to_port`

- Declare ports as **`DrivenModalPortSpec`**: `(component, pin, impedance_ohms, metadata)`. Default reference impedance is usually **50 Ω** for microwave feeds; junction pins carry Metal metadata shaped like:

```json
{"hfss_target": "junction", "draw_inductor": false}
```

so Metal draws lumped JJ boundaries correctly. **`build_*_request`** helpers embed **`port_mapping`** dicts tying logical names (**`cross`**, **`claw`**, **`feedline_input`**, **`jj`**, etc.) onto concrete component/pin targets.
- Render through **`squadds.simulations.drivenmodal.design.render_drivenmodal_design`** with **`open_pins=[]`** when providing **`port_list` / `jj_to_port`** (helper normalizes the legacy Metal concatenation guard).

Renormalization mental model:

- HFSS attaches a **reference Z** per lumped port. Feedline **`S`**-parameters quoted by Ansys inherit those references.
- When you synthesize **`S`** from a uniform **`y_to_s(..., z0_ohms=50)`** call, **every port** shares one Z0 basis — fine for symmetrical feedlines, misleading if you blindly quote JJ **`S`**.
- Prefer analyzing the qubit via **admittance** (module **`squadds.simulations.drivenmodal.qubit_admittance`**) unless you deliberately re-normalize into a surrogate circuit.

---

## Capacitance Extraction Families

### `system_kind='qubit_claw'`

- Ports: **`cross → xmon.rect_jj`**, **`claw → xmon.readout`**
- Compared pair labels: **`cross_to_ground`**, **`claw_to_ground`**, **`cross_to_claw`**, etc.
- **`capacitance_reference_summary`** scrapes dataset Q3D numbers; **`capacitance_comparison_table`** aligns driven-modal fF curves vs references.

### `system_kind='ncap'`

- Ports: **`top → cplr.prime_start`**, **`bottom → cplr.second_end`**

Before narrating capacitor topologies aloud, read **`maxwell_matrix_interpretation()`** (exposed via **`get_maxwell_capacitance_conventions`**) — mutual vs diagonal Maxwell entries confuse double counting.

---

## Coupled Quarter / Half-Wave Runs

**`build_coupled_system_request`** sets **`CoupledSystemDrivenModalRequest.resonator_type`** to **`quarter_wave`** or **`half_wave`** after normalizing textual `'quarter'/'half'`.

Ports default to **feedline start/end + JJ surrogate**. Segmented pipelines reuse geometry with **`build_segmented_coupled_system_requests`** + **`segmented_hamiltonian_sweeps`** (fine qubit sweep, coarse bridge sweep, fine cavity sweep).

Hamiltonian targets per row are summarized via **`coupled_reference_summary`** (Hamiltonian scalars plus **`transmon_state_inductances`**). Compare extracted dispersive numbers with **`hamiltonian_comparison_table`**.

---

## skrf ingestion

1. HDF / CSV exports become **`pandas`** tables indexed by GHz.
2. **`parameter_dataframe_to_tensor`** → `(freq_hz, ndarray[freq,n,n])`.
3. **`network_from_parameter_dataframe`** emits **`rf.Network`** (Touchstone-compatible) with explicit **`z0_ohms`**.
4. Some exports flatten rows—tutorials hardened parsers; replicate their column checks whenever you ingest new artifacts.

---

## When to Reach for Each Module

| Concern | Module / symbol |
|---------|----------------|
| Requests + pedagogical wrappers | **`squadds.simulations.drivenmodal.workflows`** |
| Rendering + HFSS setup guards | **`...drivenmodal.design`** |
| skrf **`Network`** + Touchstone | **`...drivenmodal.hfss_data`** |
| **`Y` reduction**, **`kappa`/`Q`**, **`g`** from **`chi`** | **`...drivenmodal.coupled_postprocess`** |
| JJ admittance hybrids + mode extraction | **`...drivenmodal.qubit_admittance`** |

---

## Suggested Agent Flow

1. Fetch **`squadds://drivenmodal-workflow`** (this prose) plus **`squadds://drivenmodal-playbook`** / **`get_drivenmodal_playbook_json`**.
2. Pull database rows via existing **`find_closest_designs`** / **`get_dataset`** MCP tools.
3. Hand off Python reproduction steps (Metal + AEDT GUI) locally—never pretend MCP substitutes HFSS batches.
"""


def register_drivenmodal_resources(mcp: FastMCP) -> None:
    """Register driven-modal MCP resources."""

    @mcp.resource("squadds://drivenmodal-workflow")
    async def drivenmodal_workflow_guide() -> str:
        """Narrative walkthrough plus API pointers for driven-modal notebooks."""
        return DRIVENMODAL_WORKFLOW_GUIDE_MARKDOWN

    @mcp.resource("squadds://drivenmodal-playbook")
    async def drivenmodal_playbook_json() -> str:
        """Structured JSON playbook mirroring MCP tool ``get_drivenmodal_playbook_json``."""
        return json.dumps(build_drivenmodal_playbook(), indent=2)
