"""
Database browsing & query tools.
================================

These tools let AI agents and humans explore the SQuADDS database:
browse supported components, list configs, load datasets (with pagination),
inspect measured devices, and view contributors.

All tools receive the ``SQuADDS_DB`` instance through the MCP lifespan context.

Adding a New Database Tool
--------------------------
1. Define an ``async def`` function with the ``@mcp.tool()`` decorator.
2. Accept ``ctx: Context`` to access ``ctx.request_context.lifespan_context.db``.
3. Return a Pydantic model (from ``schemas.py``) for structured output.
4. Register the tool in ``server.py`` → ``register_database_tools()``.
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from mcp.server.fastmcp import Context, FastMCP

from squadds_mcp.schemas import (
    ComponentListResult,
    ConfigListResult,
    DatasetInfoResult,
    DatasetResult,
    DatasetSummaryResult,
    DatasetSummaryRow,
    MeasuredDeviceResult,
)
from squadds_mcp.utils import dataframe_to_records, sanitize_for_json


def register_database_tools(mcp: FastMCP) -> None:
    """Register all database-browsing tools on the given MCP server.

    This function is called once during server startup from ``server.py``.
    Each inner function becomes an MCP tool visible to connected clients.

    Args:
        mcp: The FastMCP server instance to register tools on.
    """

    @mcp.tool()
    async def list_components(ctx: Context) -> ComponentListResult:
        """List all supported top-level component types in SQuADDS.

        Returns component types like 'qubit', 'cavity_claw', 'coupler'.
        Use these values when calling other tools that require a ``component`` parameter.
        """
        db = ctx.request_context.lifespan_context.db
        items = db.supported_components()
        return ComponentListResult(items=items, count=len(items))

    @mcp.tool()
    async def list_component_names(
        ctx: Context,
        component: str,
    ) -> ComponentListResult:
        """List available component names for a given component type.

        For example, ``list_component_names(component="qubit")`` might return
        ``["TransmonCross"]``.

        Args:
            component: A top-level component type (e.g. 'qubit', 'cavity_claw', 'coupler').
                       Get valid values from ``list_components``.
        """
        db = ctx.request_context.lifespan_context.db
        names = db.get_component_names(component)
        if names is None:
            return ComponentListResult(items=[], count=0)
        return ComponentListResult(items=names, count=len(names))

    @mcp.tool()
    async def list_configs(ctx: Context) -> ConfigListResult:
        """List all available dataset configuration names.

        Each config string has the format ``component-component_name-data_type``
        (e.g. ``qubit-TransmonCross-cap_matrix``). Use these to load specific datasets.
        """
        db = ctx.request_context.lifespan_context.db
        configs = db.configs
        return ConfigListResult(configs=configs, count=len(configs))

    @mcp.tool()
    async def list_datasets(ctx: Context) -> DatasetSummaryResult:
        """List all available datasets with their component, name, and data type.

        Provides a structured overview of every dataset in SQuADDS.
        """
        db = ctx.request_context.lifespan_context.db
        components = db.supported_components()
        component_names = db.supported_component_names()
        data_types = db.supported_data_types()

        datasets = []
        for comp, name, dtype in zip(components, component_names, data_types):
            datasets.append(
                DatasetSummaryRow(component=comp, component_name=name, data_type=dtype)
            )
        return DatasetSummaryResult(datasets=datasets, count=len(datasets))

    @mcp.tool()
    async def get_dataset_info(
        ctx: Context,
        component: str,
        component_name: str,
        data_type: str,
    ) -> DatasetInfoResult:
        """Get metadata about a specific SQuADDS dataset.

        Returns the number of rows, feature names and types, description,
        and size. Does NOT return the actual data — use ``get_dataset`` for that.

        Args:
            component: Component type (e.g. 'qubit').
            component_name: Component name (e.g. 'TransmonCross').
            data_type: Data type (e.g. 'cap_matrix', 'eigenmode').
        """
        db = ctx.request_context.lifespan_context.db
        config = f"{component}-{component_name}-{data_type}"

        dataset = load_dataset(db.repo_name, config)["train"]
        features = {name: str(feat) for name, feat in dataset.features.items()}

        return DatasetInfoResult(
            config=config,
            num_rows=len(dataset),
            features=features,
            description=dataset.info.description or "",
            size_bytes=dataset.info.size_in_bytes,
        )

    @mcp.tool()
    async def get_dataset(
        ctx: Context,
        component: str,
        component_name: str,
        data_type: str,
        limit: int = 50,
        offset: int = 0,
    ) -> DatasetResult:
        """Load a SQuADDS dataset and return rows as JSON (paginated).

        Returns up to ``limit`` rows starting at ``offset``. Large datasets
        should be paged through by incrementing ``offset``.

        Args:
            component: Component type (e.g. 'qubit').
            component_name: Component name (e.g. 'TransmonCross').
            data_type: Data type (e.g. 'cap_matrix', 'eigenmode').
            limit: Max rows per page (default 50, max 200).
            offset: Starting row index (default 0).
        """
        db = ctx.request_context.lifespan_context.db
        limit = min(limit, 200)  # Cap at 200 to protect context windows

        df = db.get_dataset(
            data_type=data_type,
            component=component,
            component_name=component_name,
        )

        if df is None:
            return DatasetResult(
                rows=[],
                total_rows=0,
                offset=offset,
                limit=limit,
                component=component,
                component_name=component_name,
                data_type=data_type,
            )

        total = len(df)
        rows = dataframe_to_records(df, limit=limit, offset=offset)

        return DatasetResult(
            rows=rows,
            total_rows=total,
            offset=offset,
            limit=limit,
            component=component,
            component_name=component_name,
            data_type=data_type,
        )

    @mcp.tool()
    async def list_measured_devices(ctx: Context) -> list[MeasuredDeviceResult]:
        """List all experimentally measured devices in SQuADDS.

        Returns device names, design codes, paper links, foundries,
        and fabrication recipes.
        """
        db = ctx.request_context.lifespan_context.db
        df = db.get_measured_devices()

        results = []
        for _, row in df.iterrows():
            results.append(
                MeasuredDeviceResult(
                    name=row.get("name", ""),
                    design_code=row.get("design_code"),
                    paper_link=row.get("paper_link"),
                    foundry=row.get("foundry"),
                    fabrication_recipe=sanitize_for_json(row.get("fabrication_recipe")),
                )
            )
        return results

    @mcp.tool()
    async def get_simulation_results(
        ctx: Context,
        device_name: str,
    ) -> dict:
        """Get simulation results for a specific measured device.

        Args:
            device_name: Name of the experimentally measured device.
        """
        db = ctx.request_context.lifespan_context.db
        results = db.view_simulation_results(device_name)
        return sanitize_for_json(results) if results else {"error": f"No results found for device '{device_name}'."}

    @mcp.tool()
    async def list_data_types(ctx: Context) -> dict:
        """List and explain available data types in SQuADDS.

        SQuADDS has two main data types:
        - **cap_matrix**: Capacitance matrix simulation data. Contains geometric
          design parameters and the resulting capacitance values from electrostatic
          simulations. Used for qubit and coupler components.
        - **eigenmode**: Eigenmode simulation data. Contains resonant frequencies,
          quality factors (kappa), and other electromagnetic properties from
          eigenmode simulations. Used for cavity_claw components.

        Use the returned data type names with ``get_dataset`` or ``get_dataset_info``.
        """
        db = ctx.request_context.lifespan_context.db
        all_types = db.supported_data_types()
        unique_types = sorted(set(all_types))

        type_descriptions = {
            "cap_matrix": {
                "description": "Capacitance matrix data from electrostatic simulations",
                "used_for": ["qubit", "coupler"],
                "typical_columns": [
                    "design_options (geometry parameters)",
                    "sim_options (simulation settings)",
                    "sim_results (capacitance matrix values)",
                    "contributor (data source info)",
                ],
                "physics": (
                    "The capacitance matrix relates voltages to charges on each island. "
                    "For a transmon: C_q determines E_C (charging energy) and thus "
                    "qubit frequency and anharmonicity. For coupled systems: cross-capacitances "
                    "determine coupling strength (g)."
                ),
            },
            "eigenmode": {
                "description": "Eigenmode simulation data (frequencies, quality factors)",
                "used_for": ["cavity_claw"],
                "typical_columns": [
                    "design_options (geometry parameters)",
                    "sim_options (simulation settings)",
                    "cavity_frequency_GHz (resonant frequency)",
                    "kappa_kHz (linewidth / coupling rate)",
                    "coupler_type (CLT for quarter-wave, NCap for half-wave)",
                ],
                "physics": (
                    "Eigenmode simulations find the natural resonant frequencies and "
                    "decay rates of the cavity. kappa (linewidth) determines the "
                    "photon lifetime and measurement speed."
                ),
            },
        }

        result = {
            "data_types": unique_types,
            "descriptions": {t: type_descriptions.get(t, {"description": t}) for t in unique_types},
        }
        return result

    @mcp.tool()
    async def get_capacitance_data(
        ctx: Context,
        component: str,
        component_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> DatasetResult:
        """Load capacitance matrix (cap_matrix) data for a component.

        This is a convenience tool that wraps ``get_dataset`` specifically
        for capacitance data, which is the most commonly needed raw data type.

        **Capacitance data is available for:**
        - Qubits: ``component='qubit'``, ``component_name='TransmonCross'``
          → Contains design geometry and resulting capacitance matrices.
          The capacitance values determine the qubit's charging energy (E_C),
          which sets the qubit frequency and anharmonicity.
        - Couplers: ``component='coupler'``, ``component_name='NCap'``
          → Contains CapNInterdigitalTee coupler geometry and capacitances.
          Used for half-wave resonator systems.

        **Note:** Cavity/resonator data uses data_type='eigenmode', not 'cap_matrix'.
        Use ``get_dataset(component='cavity_claw', ..., data_type='eigenmode')`` for that.

        Args:
            component: 'qubit' or 'coupler'.
            component_name: Component name (e.g. 'TransmonCross', 'NCap').
            limit: Max rows per page (default 50, max 200).
            offset: Starting row index (default 0).
        """
        db = ctx.request_context.lifespan_context.db
        limit = min(limit, 200)

        df = db.get_dataset(
            data_type="cap_matrix",
            component=component,
            component_name=component_name,
        )

        if df is None:
            return DatasetResult(
                rows=[],
                total_rows=0,
                offset=offset,
                limit=limit,
                component=component,
                component_name=component_name,
                data_type="cap_matrix",
            )

        total = len(df)
        rows = dataframe_to_records(df, limit=limit, offset=offset)

        return DatasetResult(
            rows=rows,
            total_rows=total,
            offset=offset,
            limit=limit,
            component=component,
            component_name=component_name,
            data_type="cap_matrix",
        )

    @mcp.tool()
    async def get_resonator_info(ctx: Context) -> dict:
        """Get detailed information about resonator types in SQuADDS.

        SQuADDS supports two resonator topologies that fundamentally affect
        the system Hamiltonian:

        **Quarter-wave (λ/4) resonators:**
        - Coupler type: CLT (Coplanar-Line-T)
        - Standard CPW resonator shorted at one end
        - More compact design
        - Most common in literature
        - Use ``resonator_type='quarter'`` in search tools

        **Half-wave (λ/2) resonators:**
        - Coupler type: NCap (CapNInterdigitalTee)
        - CPW resonator open at both ends, coupled via interdigital capacitor
        - Requires separate capacitance data for the NCap coupler
        - Different frequency-length relationship than quarter-wave
        - Use ``resonator_type='half'`` in search tools

        **Critical distinction for design searches:**
        - ``resonator_type='quarter'`` → searches CLT-coupled designs
        - ``resonator_type='half'`` → searches NCap-coupled designs
        - The resonator type affects ALL Hamiltonian parameters (frequency, kappa, g)
        - You MUST specify the correct resonator_type for accurate design matching

        This tool returns available resonator types, their coupler mappings,
        and which datasets contain data for each type.
        """
        db = ctx.request_context.lifespan_context.db

        # Check what configs exist for each type
        configs = db.configs
        has_clt = any("CLT" in c for c in configs)
        has_ncap = any("NCap" in c for c in configs)

        return {
            "resonator_types": {
                "quarter": {
                    "coupler_type": "CLT",
                    "full_name": "Quarter-wave (λ/4) resonator",
                    "coupling_mechanism": "Coplanar Line T-junction",
                    "description": (
                        "Standard CPW resonator shorted at one end. "
                        "The resonant frequency is f = c/(4*L*sqrt(ε_eff)) where L is the "
                        "total resonator length. Most common topology in superconducting circuits."
                    ),
                    "data_available": has_clt,
                    "datasets": [
                        "cavity_claw-RouteMeander-eigenmode (filtered by coupler_type='CLT')",
                        "qubit-TransmonCross-cap_matrix (qubit capacitance)",
                    ],
                    "how_to_search": {
                        "tool": "find_closest_designs",
                        "params": {"system_type": "qubit_cavity", "resonator_type": "quarter"},
                    },
                },
                "half": {
                    "coupler_type": "NCap",
                    "full_name": "Half-wave (λ/2) resonator",
                    "coupling_mechanism": "CapNInterdigitalTee (interdigital capacitor)",
                    "description": (
                        "CPW resonator open at both ends, coupled via an interdigital capacitor. "
                        "The resonant frequency is f = c/(2*L*sqrt(ε_eff)). Requires separate NCap "
                        "coupler capacitance data for accurate Hamiltonian parameter calculation."
                    ),
                    "data_available": has_ncap,
                    "datasets": [
                        "cavity_claw-RouteMeander-eigenmode (filtered by coupler_type='NCap')",
                        "qubit-TransmonCross-cap_matrix (qubit capacitance)",
                        "coupler-NCap-cap_matrix (coupler capacitance — REQUIRED for half-wave)",
                    ],
                    "how_to_search": {
                        "tool": "find_closest_designs",
                        "params": {"system_type": "qubit_cavity", "resonator_type": "half"},
                    },
                },
            },
            "key_differences": [
                "Quarter-wave resonators are more compact (shorter for same frequency)",
                "Half-wave resonators have a voltage antinode at both ends",
                "Half-wave systems require NCap coupler capacitance data for Hamiltonian extraction",
                "The coupling strength (g) depends on the coupler geometry, which differs between the two",
                "resonator_type MUST match when searching: mixing types gives incorrect results",
            ],
            "recommendation": (
                "Use 'quarter' (default) unless you specifically need half-wave resonators. "
                "Quarter-wave has more data in SQuADDS and is the standard in most qubit architectures."
            ),
        }
