"""
Contribution & reference device tools.
=======================================

Read-only tools for inspecting experimental reference devices,
fabrication recipes, and contributor metadata.

Adding a New Contribution Tool
------------------------------
1. Add your ``@mcp.tool()`` function inside ``register_contribution_tools()``.
2. These tools should remain **read-only** — no mutations to the database.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from squadds_mcp.utils import sanitize_for_json


def register_contribution_tools(mcp: FastMCP) -> None:
    """Register contribution-info tools on the MCP server.

    Args:
        mcp: The FastMCP server instance.
    """

    @mcp.tool()
    async def get_reference_device(
        ctx: Context,
        component: str,
        component_name: str,
        data_type: str,
    ) -> dict[str, Any]:
        """Get information about the reference experimental device used to validate a simulation config.

        Every simulation dataset in SQuADDS was validated against a real,
        experimentally measured device. This tool retrieves information about
        that reference device.

        Args:
            component: Component type (e.g. 'qubit').
            component_name: Component name (e.g. 'TransmonCross').
            data_type: Data type (e.g. 'cap_matrix').
        """
        db = ctx.request_context.lifespan_context.db
        info = db.get_device_contributors_of(
            component=component,
            component_name=component_name,
            data_type=data_type,
        )
        if info is None:
            return {"error": f"No reference device found for {component}-{component_name}-{data_type}."}
        return sanitize_for_json(info)

    @mcp.tool()
    async def get_fabrication_recipe(
        ctx: Context,
        device_name: str,
    ) -> dict[str, Any]:
        """Get the fabrication recipe for a specific measured device.

        Returns foundry and fabrication process details.

        Args:
            device_name: Name of the experimentally measured device.
                         Get valid names from ``list_measured_devices``.
        """
        from datasets import load_dataset

        db = ctx.request_context.lifespan_context.db
        dataset = load_dataset(db.repo_name, "measured_device_database")["train"]

        for i, entry in enumerate(dataset):
            if entry.get("name") == device_name:
                recipe_info = {
                    "device_name": device_name,
                    "foundry": entry.get("foundry", "N/A"),
                    "fabrication_recipe": sanitize_for_json(entry.get("fabrication_recipe", {})),
                }
                return recipe_info

        return {"error": f"Device '{device_name}' not found in the measured device database."}

    @mcp.tool()
    async def list_contributors(ctx: Context) -> list[dict[str, Any]]:
        """List all unique contributors to the SQuADDS simulation database.

        Returns contributor names, PIs, groups, and institutions.
        """
        from datasets import load_dataset

        db = ctx.request_context.lifespan_context.db
        seen = set()
        contributors = []

        sim_configs = [c for c in db.configs if "cap_matrix" in c or "eigenmode" in c]

        for config in sim_configs:
            try:
                dataset = load_dataset(db.repo_name, config)["train"]
                for entry in dataset:
                    contrib = entry.get("contributor", {})
                    if not isinstance(contrib, dict):
                        continue
                    key = (
                        contrib.get("uploader", ""),
                        contrib.get("PI", ""),
                        contrib.get("group", ""),
                        contrib.get("institution", ""),
                    )
                    if key not in seen:
                        seen.add(key)
                        contributors.append(sanitize_for_json(contrib))
            except Exception:
                continue

        return contributors
