"""
Tests driven-modal MCP playbook content and Maxwell helper alignment.
"""

import json

from squadds.simulations.drivenmodal.workflows import maxwell_matrix_interpretation
from squadds_mcp.drivenmodal_playbook import build_drivenmodal_playbook


class TestDrivenmodalPlaybook:
    """Smoke coverage for MCP-facing driven-modal playbook JSON."""

    def test_playbook_contains_agent_sections(self) -> None:
        data = build_drivenmodal_playbook()
        required = (
            "squadds_package",
            "workflow_phases",
            "lumped_ports_reference_impedance",
            "capacitance_extraction",
            "coupled_system_hamiltonian",
            "skrf_and_dataframe_pipeline",
            "admittance_reduction_and_loaded_response",
            "scqubits_jj_admittance_pipeline",
            "documentation_uris",
        )
        for key in required:
            assert key in data, f"missing playbook section: {key}"

    def test_playbook_documentation_consistency(self) -> None:
        data = build_drivenmodal_playbook()
        docs = data["documentation_uris"]
        assert docs["narrative_guide"] == "squadds://drivenmodal-workflow"
        assert docs["machine_playbook_mirror"] == "squadds://drivenmodal-playbook"


class TestMaxwellInterpretationLinkage:
    """Ensures MCP Maxwell helper matches library DataFrame."""

    def test_roundtrip_through_json_helpers(self) -> None:
        frame = maxwell_matrix_interpretation()
        dumped = json.dumps(frame.to_dict(orient="records"))
        rows = json.loads(dumped)
        assert len(rows) >= 2
        assert "entry" in rows[0]
