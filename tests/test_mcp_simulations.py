"""Tests unified simulation playbook (Q3D, eigenmode, driven-modal MCP surfaces)."""

import json

from squadds_mcp.simulation_playbook import (
    build_legacy_ansys_simulation_outline,
    build_simulation_playbook_full,
    build_simulation_playbook_summary,
)


class TestSimulationPlaybook:
    """Structure checks without exercising FastMCP."""

    def test_summary_has_flow_index_and_legacy_outline(self) -> None:
        data = build_simulation_playbook_summary()
        assert "flow_index" in data
        assert "legacy_ansys" in data
        ids = [entry["id"] for entry in data["flow_index"]]
        assert "q3d_capacitance_lom" in ids
        assert "hfss_eigenmode" in ids
        assert "hfss_driven_modal" in ids

    def test_full_embeddings_driven_modal_subtree_key(self) -> None:
        data = build_simulation_playbook_full()
        assert "hfss_driven_modal_playbook_nested" in data
        nested = data["hfss_driven_modal_playbook_nested"]
        assert "workflow_phases" in nested

    def test_outline_serializes_cleanly_to_json(self) -> None:
        outline = build_legacy_ansys_simulation_outline()
        dumped = json.dumps(outline)
        roundtrip = json.loads(dumped)
        assert "simulate_whole_device" in roundtrip["orchestration"]
