import os

import pytest

from squadds import Analyzer, SQuADDS_DB
from squadds.interpolations.physics import ScalingInterpolator


RUN_LIVE_TESTS = os.getenv("SQUADDS_RUN_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_LIVE_TESTS,
        reason="Set SQUADDS_RUN_LIVE_TESTS=1 to run live Hugging Face smoke tests.",
    ),
]


def test_database_metadata_queries(headless_qiskit_environment):
    db = SQuADDS_DB()

    assert "qubit" in db.supported_components()
    assert "TransmonCross" in db.get_component_names("qubit")
    assert "RouteMeander" in db.get_component_names("cavity_claw")


def test_quarter_wave_search_and_interpolation(headless_qiskit_environment):
    db = SQuADDS_DB()
    db.unselect_all()
    db.select_system(["qubit", "cavity_claw"])
    db.select_qubit("TransmonCross")
    db.select_cavity_claw("RouteMeander")
    db.select_resonator_type("quarter")

    merged_df = db.create_system_df()
    assert not merged_df.empty

    analyzer = Analyzer(db)
    target_params = {
        "qubit_frequency_GHz": 4,
        "cavity_frequency_GHz": 6.2,
        "kappa_kHz": 120,
        "resonator_type": "quarter",
        "anharmonicity_MHz": -200,
        "g_MHz": 70,
    }

    results = analyzer.find_closest(target_params=target_params, num_top=1, metric="Euclidean", display=True)
    design_df = ScalingInterpolator(analyzer, target_params).get_design()

    assert len(results) == 1
    assert not design_df.empty
    assert "design_options" in design_df.columns


def test_half_wave_search_and_interpolation(headless_qiskit_environment):
    db = SQuADDS_DB()
    db.unselect_all()
    db.select_system(["qubit", "cavity_claw"])
    db.select_qubit("TransmonCross")
    db.select_cavity_claw("RouteMeander")
    db.select_resonator_type("half")

    merged_df = db.create_system_df()
    assert not merged_df.empty

    analyzer = Analyzer(db)
    target_params = {
        "qubit_frequency_GHz": 4,
        "cavity_frequency_GHz": 6.2,
        "kappa_kHz": 120,
        "anharmonicity_MHz": -200,
        "g_MHz": 70,
    }

    results = analyzer.find_closest(target_params=target_params, num_top=1, metric="Euclidean", display=True)
    design_df = ScalingInterpolator(analyzer, target_params).get_design()

    assert len(results) == 1
    assert not design_df.empty
    assert "design_options" in design_df.columns


def test_half_wave_setup_api(headless_qiskit_environment):
    from squadds import AnsysSimulator

    db = SQuADDS_DB()
    db.unselect_all()
    db.select_system(["qubit", "cavity_claw"])
    db.select_qubit("TransmonCross")
    db.select_cavity_claw("RouteMeander")
    db.select_resonator_type("half")

    merged_df = db.create_system_df()
    assert not merged_df.empty

    analyzer = Analyzer(db)
    target_params = {
        "qubit_frequency_GHz": 4,
        "anharmonicity_MHz": -200,
        "cavity_frequency_GHz": 6.2,
        "kappa_kHz": 100,
        "g_MHz": 70,
    }

    results = analyzer.find_closest(target_params=target_params, num_top=1, metric="Euclidean", display=False)
    device = results.iloc[0]

    setup_keys = [key for key in device.keys() if "setup" in key.lower()]
    assert "setup_qubit" in setup_keys
    assert "setup_cavity_claw" in setup_keys
    assert "setup_coupler" in setup_keys

    simulator = AnsysSimulator(analyzer, device)
    all_setups = simulator.get_simulation_setup(target="all")
    cavity_setup = simulator.get_simulation_setup(target="cavity_claw")

    assert "setup_qubit" in all_setups
    assert "setup_cavity_claw" in all_setups
    assert "setup_coupler" in all_setups
    assert "setup_cavity_claw" in cavity_setup
