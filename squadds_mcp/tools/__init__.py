"""
Tool sub-package for the SQuADDS MCP server.
=============================================

Each module in this package defines a group of related MCP tools.
Tools are registered with the FastMCP server instance in ``server.py``
via the ``register_*_tools()`` functions exported from each module.

Module layout:
    - ``database.py``      — Browse datasets, configs, components, devices
    - ``analysis.py``      — Search for closest designs, compute H-params
    - ``interpolation.py`` — Physics-based scaling interpolation
    - ``contribution.py``  — View contributor info & fabrication recipes
"""
