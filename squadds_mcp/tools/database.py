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
