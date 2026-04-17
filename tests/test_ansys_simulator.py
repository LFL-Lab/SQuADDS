import json
from pathlib import Path
from unittest.mock import patch

from squadds.simulations.ansys_simulator import AnsysSimulator
from squadds.simulations.drivenmodal.models import (
    CapacitanceExtractionRequest,
    DrivenModalArtifactPolicy,
    DrivenModalLayerStackSpec,
    DrivenModalSetupSpec,
    DrivenModalSweepSpec,
)


class DummyAnalyzer:
    def __init__(self, selected_system):
        self.selected_system = selected_system


def test_get_setup_targets_and_get_simulation_setup_for_coupled_system(headless_qiskit_environment):
    simulator = AnsysSimulator(
        DummyAnalyzer(["qubit", "cavity_claw"]),
        {
            "setup_qubit": {"max_passes": 10},
            "setup_cavity_claw": {"max_passes": 11},
            "setup_coupler": {"max_passes": 12},
        },
    )

    assert simulator._get_setup_targets("all") == ["setup_qubit", "setup_cavity_claw", "setup_coupler"]
    assert simulator._get_setup_targets("cavity_claw") == ["setup_cavity_claw"]
    assert set(simulator.get_simulation_setup("all")) == {"setup_qubit", "setup_cavity_claw", "setup_coupler"}


def test_update_simulation_setup_respects_unknown_parameter_prompt(headless_qiskit_environment):
    simulator = AnsysSimulator(
        DummyAnalyzer(["qubit", "cavity_claw"]),
        {
            "setup_qubit": {"max_passes": 10},
            "setup_cavity_claw": {"max_passes": 11},
            "setup_coupler": {"max_passes": 12},
        },
    )

    with patch("builtins.input", return_value="n"):
        simulator.update_simulation_setup(target="qubit", max_passes=20, brand_new=123)

    assert simulator.device_dict["setup_qubit"]["max_passes"] == 20
    assert "brand_new" not in simulator.device_dict["setup_qubit"]


def test_normalize_device_dict_deserializes_json_like_payloads(headless_qiskit_environment):
    simulator = AnsysSimulator(
        DummyAnalyzer(["qubit", "cavity_claw"]),
        {
            "design_options_qubit": json.dumps(
                {
                    "cross_length": "200um",
                    "connection_pads": {"readout": {"claw_length": "150um", "ground_spacing": "20um"}},
                }
            ),
            "design_options_cavity_claw": json.dumps(
                {
                    "cpw_opts": {"total_length": "4000um"},
                    "cplr_opts": {"coupling_length": "250um"},
                    "claw_opts": {"connection_pads": {"readout": {"ground_spacing": "20um"}}},
                }
            ),
            "setup_cavity_claw_merged": json.dumps({"setup": {"max_passes": 15}}),
        },
    )

    assert simulator.device_dict["design_options_qubit"]["cross_length"] == "200um"
    assert simulator.device_dict["design_options_cavity_claw"]["cpw_opts"]["total_length"] == "4000um"
    assert simulator.device_dict["setup_cavity_claw"] == {"max_passes": 15}


def test_run_drivenmodal_initializes_checkpoint_manifest(headless_qiskit_environment, tmp_path: Path):
    simulator = AnsysSimulator(
        DummyAnalyzer("qubit_claw"),
        {
            "design_options": {"cross_length": "200um"},
            "setup": {"freq_ghz": 5.0},
        },
    )
    request = CapacitanceExtractionRequest(
        system_kind="qubit_claw",
        design_payload={"design_options": {"cross_length": "200um"}},
        layer_stack=DrivenModalLayerStackSpec(),
        setup=DrivenModalSetupSpec(),
        sweep=DrivenModalSweepSpec(start_ghz=1.0, stop_ghz=10.0, count=101),
        artifacts=DrivenModalArtifactPolicy(),
        metadata={"run_id": "qubit-claw-demo"},
    )

    result = simulator.run_drivenmodal(request, checkpoint_dir=tmp_path)

    assert result["request"]["system_kind"] == "qubit_claw"
    assert result["manifest"]["run_id"] == "qubit-claw-demo"
    assert (tmp_path / "qubit-claw-demo" / "manifest.json").exists()
