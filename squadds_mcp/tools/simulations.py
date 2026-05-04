"""
MCP tooling for documenting *all* SQuADDS Ansys-class workflows:
legacy Q3D capacitance, HFSS eigenmode, sweeps, and HFSS driven-modal notebooks.

Execution remains on the user's machine; JSON returned here is reference material.
"""

from __future__ import annotations

import json
from typing import Literal

from mcp.server.fastmcp import FastMCP

from squadds.simulations.drivenmodal.workflows import maxwell_matrix_interpretation
from squadds_mcp.drivenmodal_playbook import build_drivenmodal_playbook
from squadds_mcp.simulation_playbook import (
    build_simulation_playbook_full,
    build_simulation_playbook_summary,
)
from squadds_mcp.utils import sanitize_for_json

PlaybookVariant = Literal["summary", "full"]

_VALID_VARIANTS: tuple[PlaybookVariant, ...] = ("summary", "full")


def register_simulation_tools(mcp: FastMCP) -> None:
    """Register unified simulation playbook + granular helpers shared by tutorials."""

    @mcp.tool()
    async def get_squadds_simulation_playbook(playbook_variant: PlaybookVariant) -> str:
        """Return JSON playbook for legacy Q3D, HFSS eigenmode, sweeps + driven-modal.

        Args:
            playbook_variant:
                - "summary": compact unified outline + legacy flow map;
                  matches resource `squadds://simulation-playbook-summary`.
                - "full": summary PLUS nested driven-modal playbook object under
                  `hfss_driven_modal_playbook_nested`.
        """
        if playbook_variant not in _VALID_VARIANTS:
            allowed = ", ".join(_VALID_VARIANTS)
            raise ValueError(f"Invalid playbook_variant. Choose one of: {allowed}")
        if playbook_variant == "summary":
            return json.dumps(build_simulation_playbook_summary(), indent=2)
        return json.dumps(build_simulation_playbook_full(), indent=2)

    @mcp.tool()
    async def get_drivenmodal_playbook_json() -> str:
        """Structured JSON playbook for HFSS driven-modal workflows in SQuADDS.

        Prefer `get_squadds_simulation_playbook(playbook_variant='full')` when you
        want driven-modal folded next to classical Q3D/eigenmode context.

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
