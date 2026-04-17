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


def _print_section(title, value):
    print("=" * 80)
    print(title)
    print(value)


def test_database_metadata_queries(headless_qiskit_environment):
    db = SQuADDS_DB()
    supported_components = db.supported_components()
    qubit_names = db.get_component_names("qubit")
    cavity_names = db.get_component_names("cavity_claw")

    _print_section("Supported components", supported_components)
    _print_section("Qubit component names", qubit_names)
    _print_section("Cavity component names", cavity_names)

    assert "qubit" in supported_components
    assert "TransmonCross" in qubit_names
    assert "RouteMeander" in cavity_names


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

    _print_section("Quarter-wave merged dataframe preview", merged_df.head())
    _print_section("Quarter-wave closest results", results)
    _print_section("Quarter-wave interpolated design", design_df)

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

    _print_section("Half-wave merged dataframe preview", merged_df.head())
    _print_section("Half-wave closest results", results)
    _print_section("Half-wave interpolated design", design_df)

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
    assert any(
        key in setup_keys for key in ["setup_cavity_claw", "setup_cavity_claw_merged", "setup_cavity_claw_closest"]
    )
    assert "setup_coupler" in setup_keys

    simulator = AnsysSimulator(analyzer, device)
    all_setups = simulator.get_simulation_setup(target="all")
    cavity_setup = simulator.get_simulation_setup(target="cavity_claw")

    _print_section("Half-wave closest device", results)
    _print_section("Half-wave setup keys", setup_keys)
    _print_section("Half-wave all setups", all_setups)
    _print_section("Half-wave cavity setup", cavity_setup)

    assert "setup_qubit" in all_setups
    assert "setup_cavity_claw" in all_setups
    assert "setup_coupler" in all_setups
    assert "setup_cavity_claw" in cavity_setup
