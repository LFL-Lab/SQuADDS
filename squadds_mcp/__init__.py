"""
SQuADDS MCP Server
==================

Model Context Protocol (MCP) server for the SQuADDS
(Superconducting Qubit And Device Design and Simulation) database.

This package provides a thin MCP wrapper around the SQuADDS Python library,
exposing database queries, design search, and interpolation capabilities
to AI agents and human developers via the standardized MCP protocol.

Usage:
    # Run via CLI
    squadds-mcp

    # Or run via uv
    uv run squadds-mcp

    # Or import and run programmatically
    from squadds_mcp.server import create_server
    server = create_server()
    server.run(transport="stdio")
"""

__version__ = "0.1.0"
