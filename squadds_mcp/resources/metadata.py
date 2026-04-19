"""
Metadata resources.
====================

MCP Resources that expose read-only metadata about the SQuADDS database.
Unlike tools, resources cannot take arbitrary parameters — they are
addressed by URI and return static or semi-static data.

Adding a New Resource
---------------------
1. Add a ``@mcp.resource("squadds://your_uri")`` decorated function
   inside ``register_metadata_resources()``.
2. Resources should be cheap and fast — no heavy computation.
3. Return a string (plain text or JSON).
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context, FastMCP


def register_metadata_resources(mcp: FastMCP) -> None:
    """Register all metadata resources on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.resource("squadds://version")
    async def get_version(ctx: Context) -> str:
        """SQuADDS and MCP server version information."""
        import squadds
        from squadds_mcp import __version__ as mcp_version

        return json.dumps(
            {
                "squadds_version": squadds.__version__,
                "mcp_server_version": mcp_version,
                "python_package": "SQuADDS",
                "repo": "https://github.com/LFL-Lab/SQuADDS",
                "huggingface": "https://huggingface.co/datasets/SQuADDS/SQuADDS_DB",
            },
            indent=2,
        )

    @mcp.resource("squadds://citation")
    async def get_citation() -> str:
        """BibTeX citation for SQuADDS. Please cite this when using SQuADDS results."""
        return """@article{Shanto2024squaddsvalidated,
  doi = {10.22331/q-2024-09-09-1465},
  url = {https://doi.org/10.22331/q-2024-09-09-1465},
  title = {{SQ}u{ADDS}: {A} validated design database and simulation workflow for superconducting qubit design},
  author = {Shanto, Sadman and Kuo, Andre and Miyamoto, Clark and Zhang, Haimeng and Maurya, Vivek and Vlachos, Evangelos and Hecht, Malida and Shum, Chung Wa and Levenson-Falk, Eli},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F\\"orderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {8},
  pages = {1465},
  month = sep,
  year = {2024}
}"""

    @mcp.resource("squadds://components")
    async def get_components(ctx: Context) -> str:
        """JSON list of all supported top-level component types."""
        db = ctx.request_context.lifespan_context.db
        return json.dumps(db.supported_components(), indent=2)

    @mcp.resource("squadds://configs")
    async def get_configs(ctx: Context) -> str:
        """JSON list of all dataset configuration strings."""
        db = ctx.request_context.lifespan_context.db
        return json.dumps(db.configs, indent=2)

    @mcp.resource("squadds://datasets")
    async def get_datasets_summary(ctx: Context) -> str:
        """Summary table of all available datasets (component, name, data type)."""
        db = ctx.request_context.lifespan_context.db
        components = db.supported_components()
        component_names = db.supported_component_names()
        data_types = db.supported_data_types()

        datasets = []
        for c, n, d in zip(components, component_names, data_types):
            datasets.append({"component": c, "component_name": n, "data_type": d})

        return json.dumps(datasets, indent=2)

    @mcp.resource("squadds://guide")
    async def get_usage_guide() -> str:
        """Quick reference guide for AI agents using the SQuADDS MCP server.

        This resource provides a concise overview of the available tools
        and the recommended workflow for common tasks.
        """
        return """# SQuADDS MCP Server — Quick Reference

## What is SQuADDS?
SQuADDS (Superconducting Qubit And Device Design and Simulation) is a database
of pre-simulated superconducting quantum device designs. Given target Hamiltonian
parameters, it finds the closest matching design geometries.

## Common Workflows

### 1. Explore the Database
   - `list_components` → see component types (qubit, cavity_claw, coupler)
   - `list_component_names(component="qubit")` → see qubit types
   - `list_datasets` → see all available datasets
   - `list_data_types` → understand cap_matrix vs eigenmode data
   - `get_resonator_info` → understand quarter vs half wave resonators
   - `get_dataset_info(component, name, data_type)` → dataset metadata

### 2. Find a Design (Most Common)
   - `get_hamiltonian_param_keys(system_type)` → discover valid target params
   - `find_closest_designs(system_type, target_params, ...)` → search!
   - Result includes design_options (geometry) + hamiltonian_params (physics)
   - **IMPORTANT**: Set `resonator_type` correctly ('quarter' or 'half')

### 3. Get an Interpolated Design
   - `interpolate_design(target_params, ...)` → physics-scaled design
   - Only for qubit_cavity systems

### 4. Query Capacitance Data
   - `get_capacitance_data(component="qubit", component_name="TransmonCross")` → qubit cap matrix
   - `get_capacitance_data(component="coupler", component_name="NCap")` → NCap coupler caps
   - Capacitance determines E_C (charging energy) → qubit freq + anharmonicity

### 5. Inspect Reference Devices
   - `list_measured_devices` → all experimental devices
   - `get_reference_device(component, name, data_type)` → validation source
   - `get_fabrication_recipe(device_name)` → how to build it

## System Types
   - `qubit` — standalone qubit (params: qubit_frequency_GHz, anharmonicity_MHz)
   - `cavity_claw` — standalone cavity (params: cavity_frequency_GHz, kappa_kHz)
   - `qubit_cavity` — coupled system (all params above + g_MHz)

## Resonator Types (CRITICAL for coupled systems)
   - `quarter` (default) — λ/4 resonator with CLT coupler. Most common.
   - `half` — λ/2 resonator with NCap (interdigital capacitor) coupler.
   - The resonator type affects ALL Hamiltonian parameters!
   - Always specify the correct type when searching for designs.
   - Call `get_resonator_info` for full details.

## Data Types
   - `cap_matrix` — Capacitance matrix data (for qubits and couplers).
     Contains design geometry → capacitance values.
   - `eigenmode` — Eigenmode simulation data (for cavities).
     Contains design geometry → frequency, kappa, coupler_type.
   - Call `list_data_types` for full explanations.

## Typical Target Parameters
   - qubit_frequency_GHz: 3–8 GHz
   - anharmonicity_MHz: −500 to −50 MHz
   - cavity_frequency_GHz: 5–12 GHz
   - kappa_kHz: 10–1000 kHz
   - g_MHz: 10–200 MHz
   - resonator_type: "quarter" or "half"

## Chip Layout and Design

### Layout Best Practices
Read `squadds://layout-guide` BEFORE generating any qiskit-metal layout code.
It covers trace width matching, fillet rules, meander kink fixes, impedance
matching, and charge line routing.

### Fab-Ready Chip Design
Read `squadds://chip-design-reference` for the complete workflow from
SQuADDS search results to a GDS file ready for fabrication. Use the
`design_fab_ready_chip` prompt for a guided step-by-step walkthrough.

### DO NOT Use QubitCavity for Production
The `QubitCavity` class (`squadds.components.coupled_systems`) is a
convenience wrapper for quick visualization only. It has known geometry
issues (trace width mismatches, uncontrolled kinks). For any real design,
build each component individually using qiskit-metal primitives as
described in the layout guide and chip design reference.
"""
