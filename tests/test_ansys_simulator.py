from unittest.mock import patch

from squadds.simulations.ansys_simulator import AnsysSimulator


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
