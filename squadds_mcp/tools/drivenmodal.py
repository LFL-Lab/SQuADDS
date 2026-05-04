"""
MCP tooling for documenting HFSS driven-modal workflows shipped with SQuADDS.

These tools surface structured JSON/teaching aides so MCP-connected agents obtain
parity with repo tutorials without implying remote HFSS execution.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from squadds.simulations.drivenmodal.workflows import maxwell_matrix_interpretation
from squadds_mcp.drivenmodal_playbook import build_drivenmodal_playbook
from squadds_mcp.utils import sanitize_for_json


def register_drivenmodal_tools(mcp: FastMCP) -> None:
    """Register driven-modal explanatory tools."""

    @mcp.tool()
    async def get_drivenmodal_playbook_json() -> str:
        """Structured JSON playbook for HFSS driven-modal workflows in SQuADDS.

        Use this BEFORE attempting to mimic Tutorial-10/Tutorial-11 driven-modal notebooks.
        It enumerates lumped-port conventions, capacitor pair labels, segmented sweeps,
        skrf ingestion (`network_from_parameter_dataframe`), admittance reductions, and scqubits touchpoints.

        Returns:
            Indented JSON string from ``build_drivenmodal_playbook()``.
        """
        return json.dumps(build_drivenmodal_playbook(), indent=2)

    @mcp.tool()
    async def get_maxwell_capacitance_conventions() -> str:
        """Table describing Maxwell-matrix interpretation for Tutorial-10 style capacitances.

        Prevents incorrect double-counting when agents translate HFSS Maxwell matrices into pair capacitors.

        Returns:
            JSON list of row dicts sourced from ``maxwell_matrix_interpretation()`` DataFrame rows.
        """
        frame = maxwell_matrix_interpretation()
        records = sanitize_for_json(frame.to_dict(orient="records"))
        return json.dumps(
            {"rows": records, "source": "squadds.simulations.drivenmodal.workflows.maxwell_matrix_interpretation"},
            indent=2,
        )
