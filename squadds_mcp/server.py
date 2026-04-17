"""
SQuADDS MCP Server — Main entrypoint.
======================================

This module creates and configures the FastMCP server instance,
registers all tools/resources/prompts, and provides the CLI entrypoint.

Architecture:
    ┌─────────────┐
    │  AI Client   │  (Claude Desktop, Cursor, custom agent)
    └──────┬──────┘
           │ MCP Protocol (stdio or HTTP)
    ┌──────▼──────┐
    │  FastMCP     │  ← This module
    │  Server      │
    ├─────────────┤
    │  Tools       │  database.py, analysis.py, interpolation.py, contribution.py
    │  Resources   │  metadata.py
    │  Prompts     │  workflows.py
    ├─────────────┤
    │  Lifespan    │  Initializes SQuADDS_DB once at startup
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  SQuADDS     │  Core library (database, analyzer, interpolator)
    │  Library     │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ HuggingFace  │  Dataset storage
    └─────────────┘

Running the Server:
    # Via CLI (after uv sync --extra mcp)
    squadds-mcp

    # Via uv directly
    uv run squadds-mcp

    # With HTTP transport
    SQUADDS_MCP_TRANSPORT=streamable-http squadds-mcp

    # Programmatically
    from squadds_mcp.server import create_server
    server = create_server()
    server.run(transport="stdio")
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Lifespan context — holds shared resources across all tool invocations
# ---------------------------------------------------------------------------


@dataclass
class SQuADDSContext:
    """Application context shared across all MCP tool/resource calls.

    Attributes:
        db: The initialized SQuADDS_DB singleton instance.
             All tools access the database through this object.

    Developer note:
        To add new shared resources (e.g. a cache, a connection pool),
        add them as fields here and initialize them in ``app_lifespan()``.
    """

    db: object  # squadds.core.db.SQuADDS_DB


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[SQuADDSContext]:
    """Manage the SQuADDS MCP server lifecycle.

    This runs once when the server starts and yields a context object
    that is available to all tools via ``ctx.request_context.lifespan_context``.

    What it does:
        1. Imports SQuADDS_DB (heavy import — does HuggingFace API calls)
        2. Initializes the singleton DB instance
        3. Yields it as shared context
        4. Cleans up on shutdown (currently no-op)
    """
    # Import here to keep module-level imports lightweight.
    # SQuADDS_DB.__init__ fetches config info from HuggingFace.
    from squadds.core.db import SQuADDS_DB

    db = SQuADDS_DB()
    try:
        yield SQuADDSContext(db=db)
    finally:
        # Cleanup if needed in the future
        pass


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server() -> FastMCP:
    """Create and configure the SQuADDS MCP server.

    This function:
        1. Creates a FastMCP server instance with lifespan management
        2. Registers all tools from the tools sub-package
        3. Registers all resources from the resources sub-package
        4. Registers all prompts from the prompts sub-package

    Returns:
        A fully configured FastMCP server instance, ready to run.

    Example:
        >>> server = create_server()
        >>> server.run(transport="stdio")
    """
    mcp = FastMCP(
        name="SQuADDS",
        instructions=(
            "SQuADDS MCP Server — Access the Superconducting Qubit And Device "
            "Design and Simulation Database. Use this server to search for "
            "quantum device designs matching target Hamiltonian parameters, "
            "explore the database, and get interpolated designs. "
            "Start by reading the 'squadds://guide' resource for a quick overview."
        ),
        lifespan=app_lifespan,
    )

    # -- Register tools (grouped by domain) --
    from squadds_mcp.tools.analysis import register_analysis_tools
    from squadds_mcp.tools.contribution import register_contribution_tools
    from squadds_mcp.tools.database import register_database_tools
    from squadds_mcp.tools.interpolation import register_interpolation_tools

    register_database_tools(mcp)
    register_analysis_tools(mcp)
    register_interpolation_tools(mcp)
    register_contribution_tools(mcp)

    # -- Register resources --
    from squadds_mcp.resources.metadata import register_metadata_resources

    register_metadata_resources(mcp)

    # -- Register prompts --
    from squadds_mcp.prompts.workflows import register_workflow_prompts

    register_workflow_prompts(mcp)

    return mcp


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the SQuADDS MCP server.

    Called by the ``squadds-mcp`` console script (defined in pyproject.toml).

    Reads transport configuration from environment variables:
        - SQUADDS_MCP_TRANSPORT: "stdio" (default) or "streamable-http"
        - SQUADDS_MCP_HOST: HTTP host (default "0.0.0.0")
        - SQUADDS_MCP_PORT: HTTP port (default 8000)
    """
    transport = os.environ.get("SQUADDS_MCP_TRANSPORT", "stdio")
    server = create_server()

    if transport == "streamable-http":
        host = os.environ.get("SQUADDS_MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("SQUADDS_MCP_PORT", "8000"))
        server.run(transport="streamable-http", host=host, port=port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
