"""
Workflow prompt templates.
==========================

These prompts guide AI agents through common SQuADDS workflows.
Each prompt returns a string that the AI agent uses as instructions
for a multi-step interaction.

Adding a New Prompt
-------------------
1. Add a ``@mcp.prompt()`` decorated function inside
   ``register_workflow_prompts()``.
2. The function should return a string with step-by-step instructions.
3. Reference specific tool names so the agent knows what to call.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def register_workflow_prompts(mcp: FastMCP) -> None:
    """Register all workflow prompts on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.prompt()
    def design_qubit_cavity(
        qubit_frequency: float = 4.0,
        anharmonicity: float = -200.0,
        cavity_frequency: float = 9.2,
        kappa: float = 80.0,
        coupling_g: float = 70.0,
        resonator_type: str = "quarter",
    ) -> str:
        """Step-by-step workflow for designing a coupled qubit-cavity system.

        Guides the AI agent through the complete design process:
        finding closest designs, reviewing parameters, and optionally
        interpolating for a better match.
        """
        return f"""# Design a Qubit-Cavity System with SQuADDS

## Target Parameters
- Qubit frequency: {qubit_frequency} GHz
- Anharmonicity: {anharmonicity} MHz
- Cavity frequency: {cavity_frequency} GHz
- Kappa: {kappa} kHz
- Coupling strength (g): {coupling_g} MHz
- Resonator type: {resonator_type}

## Step-by-Step Workflow

### Step 1: Verify available components
Call `list_components` to see what component types are available.
Then call `list_component_names(component="qubit")` and
`list_component_names(component="cavity_claw")` to see specific options.

### Step 2: Find closest pre-simulated designs
Call `find_closest_designs` with:
- system_type: "qubit_cavity"
- target_params: {{
    "qubit_frequency_GHz": {qubit_frequency},
    "anharmonicity_MHz": {anharmonicity},
    "cavity_frequency_GHz": {cavity_frequency},
    "kappa_kHz": {kappa},
    "g_MHz": {coupling_g},
    "resonator_type": "{resonator_type}"
  }}
- num_results: 3

### Step 3: Review results
Examine the returned designs. For each design, check:
- How close the Hamiltonian parameters are to the targets
- The design_options (geometry: cross_length, claw_length, total_length, etc.)
- The coupler_type

### Step 4 (Optional): Get an interpolated design
If the closest design isn't close enough, call `interpolate_design` with
the same target_params to get a physics-scaled design that better matches.

### Step 5: Report the design
Summarize the best design with:
- Qubit geometry (cross_length, cross_gap, claw_length, etc.)
- Cavity geometry (total_length, coupling_length, etc.)
- Expected Hamiltonian parameters
- Josephson inductance (Lj) needed for fabrication
"""

    @mcp.prompt()
    def explore_database() -> str:
        """Guide for exploring the SQuADDS database structure and contents.

        Walks through discovering components, datasets, and measured devices.
        """
        return """# Explore the SQuADDS Database

## Overview
SQuADDS is a validated database of superconducting quantum device designs.
Follow these steps to understand what's available.

## Step 1: See what's in the database
- Call `list_components` to see component types (qubit, cavity_claw, coupler)
- Call `list_datasets` to see every available dataset

## Step 2: Understand a specific dataset
- Call `get_dataset_info(component, component_name, data_type)` for metadata
- Call `get_dataset(component, component_name, data_type, limit=5)` to preview rows

## Step 3: Explore measured devices
- Call `list_measured_devices` to see experimental devices
- Call `get_simulation_results(device_name)` to see validation data
- Call `get_fabrication_recipe(device_name)` for fabrication details

## Step 4: Check contributors
- Call `list_contributors` to see who contributed simulation data

## Key Concepts
- **Components**: qubit, cavity_claw, coupler — the building blocks
- **Component names**: specific implementations (e.g. TransmonCross, RouteMeander)
- **Data types**: cap_matrix (capacitance) or eigenmode (frequency/kappa)
- **Config strings**: "component-name-data_type" (e.g. "qubit-TransmonCross-cap_matrix")
- **Measured devices**: experimentally validated chips used to calibrate simulations
"""

    @mcp.prompt()
    def find_optimal_design(
        parameter_description: str = "a qubit at 5 GHz with low anharmonicity",
    ) -> str:
        """Workflow for finding and comparing designs based on a natural-language specification.

        This prompt helps translate natural-language requirements into
        specific Hamiltonian parameters and then search for matching designs.
        """
        return f"""# Find the Optimal Design

## User Requirement
"{parameter_description}"

## Step 1: Translate to Hamiltonian parameters
Based on the user's description, determine appropriate values for:
- qubit_frequency_GHz (typical: 3–8 GHz)
- anharmonicity_MHz (typical: −500 to −50 MHz)
- cavity_frequency_GHz (typical: 5–12 GHz, if cavity is needed)
- kappa_kHz (typical: 10–1000 kHz, if cavity is needed)
- g_MHz (typical: 10–200 MHz, if coupled system)
- resonator_type: "quarter" (standard) or "half"

Call `get_hamiltonian_param_keys` to confirm valid keys for the system type.

## Step 2: Search for designs
Use `find_closest_designs` with the translated parameters.
Start with num_results=5 to compare options.

## Step 3: Compare candidates
For each result, extract and compare:
- Hamiltonian parameters vs. targets (how close?)
- Design geometry (feasibility, size constraints)
- Coupler type (CLT = quarter-wave, NCap = half-wave)

## Step 4: Refine if needed
If no design is close enough:
- Try `interpolate_design` for a scaled version
- Adjust target parameters slightly and re-search
- Try a different resonator_type

## Step 5: Present the recommendation
Report the best design with clear geometry specs and expected performance.
"""
