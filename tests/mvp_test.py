import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
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


# QCap user contract: the calls users make, kept backwards compatible across versions (needs the [ml] extra).

QCAP_GDS = Path(__file__).parent / "data" / "capml" / "xmon_exp13_0032.gds"


def _assert_maxwell(names, C):
    C = np.asarray(C)
    assert names[0] == "G" and C.shape == (len(names), len(names))
    np.testing.assert_allclose(C, C.T, rtol=0, atol=1e-9)
    assert np.all(C[~np.eye(len(names), dtype=bool)] < 0)


def test_qcap_model_names():
    pytest.importorskip("torch")
    from squadds.ml.capacitance import (  # noqa: F401
        CapacitancePredictor,
        PlanarCapacitanceDB,
        load_model,
        optimized_physics_prior,
        physics_prior,
    )

    assert load_model().name == "QCap"
    for name in ("QCap", "qcap", "QCAP"):
        assert load_model(name).name == "QCap"
    for name in ("QCap-Fast", "qcap-fast", "QCap_Fast", "qcap_fast"):
        assert load_model(name).name == "QCap-Fast"
    with pytest.raises(ValueError, match="QCap"):
        load_model("flagship")


@pytest.mark.parametrize("name", ["QCap", "QCap-Fast"])
def test_qcap_predicts_maxwell_matrix(name):
    pytest.importorskip("torch")
    from squadds.ml.capacitance import CapacitancePredictor, load_model

    res = CapacitancePredictor(load_model(name)).predict(QCAP_GDS)
    _print_section(f"{name} on {QCAP_GDS.name}", res)
    _assert_maxwell(res.names, res.matrix_fF)
    assert res.model == name and res.checks["passed"] and res.timing_s["total_s"] > 0

    sim = res.to_sim_results()
    signal = res.names[1:]
    expected = {"ground_to_ground", "maxwell_matrix", "units"}
    expected |= {f"{a}_to_{a}" for a in signal} | {f"{a}_to_ground" for a in signal}
    expected |= {f"{a}_to_{b}" for i, a in enumerate(signal) for b in signal[i + 1 :]}
    assert set(sim) == expected and sim["units"] == "fF"
    assert set(sim["maxwell_matrix"]) == {f"C_{a}_{b}" for a in res.names for b in res.names}


def test_qcap_physics_prior():
    pytest.importorskip("torch")
    from squadds.ml.capacitance import load_model, physics_prior
    from squadds.ml.capacitance.predictor import CapacitancePredictor

    prior = physics_prior(QCAP_GDS)
    assert prior.names == CapacitancePredictor(load_model("QCap-Fast")).predict(QCAP_GDS).names
    _assert_maxwell(prior.names, prior.C_fF)


def test_planar_capacitance_data_sources(headless_qiskit_environment):
    from squadds.ml.capacitance import PlanarCapacitanceDB

    designs = PlanarCapacitanceDB().designs()
    assert len(designs) > 0
    assert {"design_id", "layout_id", "family", "n_conductors", "split", "squadds_config"} <= set(designs.columns)

    assert SQuADDS_DB().enabled_sources() == ["default", "planar_capacitance"]
    try:
        assert SQuADDS_DB(sources=("default",)).enabled_sources() == ["default"]
    finally:
        SQuADDS_DB()  # back to the default sources for the other tests


def test_qcap_hosted_api():
    from squadds.ml import client

    assert {"qcap", "qcap-fast"} <= {m["id"] for m in client.list_models(timeout=120)}
    out = client.predict("qcap", file=QCAP_GDS, timeout=600)["outputs"]
    _assert_maxwell(out["names"], out["matrix_fF"])


def test_qcap_cli():
    pytest.importorskip("torch")
    exe = shutil.which("squadds-ml") or str(Path(sys.executable).parent / "squadds-ml")
    run = subprocess.run([exe, "predict", str(QCAP_GDS)], capture_output=True, text=True, timeout=600)
    _print_section("squadds-ml predict", run.stdout)
    assert run.returncode == 0, run.stderr
    rows = [line for line in run.stdout.splitlines() if line and not line.startswith("#")]
    assert rows[0].split(",")[1:] == ["G", "C1", "C2", "C3"] and len(rows) == 5
