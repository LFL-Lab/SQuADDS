"""MLSimulator / Analyzer.predict_capacitance / calcs helpers (CI-safe: a stand-in predictor, no model download)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gdstk")
pytest.importorskip("klayout.db")

from squadds.ml.capacitance.bridge import (  # noqa: E402
    EC_MHz,
    EC_MHz_from_matrix,
    MLSimulator,
    build_component,
    g_MHz,
    predict_dataframe,
    transmon_cross_capacitances,
)
from squadds.ml.capacitance.result import CapacitanceResult  # noqa: E402

XMON = {
    "cross_length": "275.972um",
    "cross_width": "22.373um",
    "cross_gap": "36.448um",
    "connection_pads": {
        "readout": {
            "connector_location": "90",
            "connector_type": "0",
            "claw_length": "199.628um",
            "claw_width": "15.441um",
            "claw_gap": "6.295um",
            "ground_spacing": "6.849um",
            "claw_cpw_length": "49.099um",
            "claw_cpw_width": "10um",
        }
    },
}


class AreaPredictor:
    """Deterministic stand-in: mutual capacitance from conductor areas, in the CapacitanceResult shape."""

    model = SimpleNamespace(name="QCap")

    def __init__(self):
        self.calls = 0

    def predict(self, lay, **kwargs):
        self.calls += 1
        areas = np.array([1e-3 * g.area for g in lay.conductors])
        n = len(areas) + 1
        M = np.zeros((n, n))
        M[0, 1:] = M[1:, 0] = areas
        for i in range(1, n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = 0.05 * min(areas[i - 1], areas[j - 1])
        C = np.diag(M.sum(1)) - M
        return CapacitanceResult(
            names=["G", *lay.names],
            matrix_fF=C,
            model="QCap",
            timing_s={"total_s": 0.01},
            substrate={"eps_r": 11.45, "thickness_um": 350.0},
        )


@pytest.fixture
def sim(headless_qiskit_environment):
    return MLSimulator(predictor=AreaPredictor())


def test_interdigital_component_prediction_has_squadds_keys(sim):
    comp = build_component({"design_options": {}}, component_name="GeneralizedCapNInterdigital")
    res = sim.predict(comp)
    sr = res["sim_results"]
    assert {"north_to_south", "north_to_ground", "south_to_ground", "top_to_bottom", "ground_to_ground"} <= set(sr)
    assert sr["units"] == "fF" and sr["maxwell_matrix"]["C_N_S"] < 0
    assert sr["north_to_south"] == pytest.approx(-sr["maxwell_matrix"]["C_N_S"])
    assert res["fidelity"] == "ml:QCap" and res["timing"]["render_s"] > 0 and "total_s" in res["timing"]
    assert set(res) >= {"design", "sim_options", "sim_results", "prediction"}
    assert res["net_names"] == {"G": "ground", "N": "north", "S": "south"}


def test_simulate_takes_an_ansys_simulator_device_dict(sim):
    analyzer = SimpleNamespace(selected_system="qubit", selected_coupler=None, selected_component_name="TransmonCross")
    s = MLSimulator(analyzer, {"design_options": XMON, "setup": {"name": "sweep"}}, predictor=AreaPredictor())
    res = s.simulate()
    sr = res["sim_results"]
    assert {"cross_to_claw", "cross_to_ground", "claw_to_ground", "cross_to_cross", "claw_to_claw"} <= set(sr)
    assert res["design"]["component_class"] == "TransmonCross"
    c = transmon_cross_capacitances(res)
    assert c["cross_to_claw"] == pytest.approx(sr["cross_to_claw"])
    two = {
        **XMON,
        "connection_pads": {
            **XMON["connection_pads"],
            "bus": {**XMON["connection_pads"]["readout"], "connector_location": "270"},
        },
    }
    res2 = sim.predict({"design": {"design_options": two, "component_class": "Xmon"}})
    assert res2["sim_results"]["cross_to_bus_claw"] > 0
    with pytest.raises(NotImplementedError):
        MLSimulator(SimpleNamespace(selected_system=["qubit", "cavity_claw"]), {}, predictor=AreaPredictor()).simulate()


def test_gds_path_input_keeps_predictor_names(sim, tmp_path):
    comp = build_component({"design_options": {}}, component_name="GeneralizedCapConcentric")
    from squadds.ml.capacitance.bridge import render_gds

    gds = render_gds(comp.design, tmp_path / "c.gds")
    res = sim.predict(str(gds))
    assert "C1_to_C2" in res["sim_results"] and res["design"]["gds"] == str(gds)
    named = sim.predict(comp)
    assert {"north_to_south", "south_to_ground"} <= set(named["sim_results"])


def test_predict_dataframe_and_analyzer_hook(sim):
    from squadds.core.analysis import Analyzer

    rows = pd.DataFrame(
        {
            "design_options": [{}, {"finger_count": "3"}],
            "coupler_type": ["GeneralizedCapNInterdigital"] * 2,
            "north_to_south": [1.0, 2.0],
        },
        index=[7, 9],
    )
    out = predict_dataframe(rows, simulator=sim)
    assert list(out.index) == [7, 9] and (out.fidelity == "ml:QCap").all()
    assert out.north_to_south_ml.iloc[0] != out.north_to_south_ml.iloc[1]
    keys = (
        "selected_component_name",
        "selected_component",
        "selected_data_type",
        "selected_confg",
        "selected_qubit",
        "selected_cavity",
        "selected_resonator_type",
        "selected_coupler",
        "selected_df",
        "qubit_df",
        "cavity_df",
        "coupler_df",
    )
    db = SimpleNamespace(selected_system="coupler", **dict.fromkeys(keys))
    analyzer = Analyzer(db)
    analyzer.closest_df = rows
    via = analyzer.predict_capacitance(simulator=sim)
    pd.testing.assert_series_equal(via.north_to_south_ml, out.north_to_south_ml)


def test_export_choice_is_recorded(sim, tmp_path):
    from squadds.layouts.dataset_export import export_dataset_gds
    from squadds.ml.capacitance.qc import geom_hash

    comp = build_component({"design_options": {}}, component_name="GeneralizedCapNInterdigital")
    res = MLSimulator(predictor=AreaPredictor(), workdir=tmp_path).predict(comp)
    exp = res["export"]
    assert exp["convention"] == "dataset" and exp["family"] == "interdigital" and exp["margin_gaps"] == 20.0
    assert res["design"]["export"] == exp
    assert geom_hash(res["design"]["gds"]) == geom_hash(export_dataset_gds(comp, tmp_path / "ref.gds"))
    assert {"north_to_south", "north_to_ground"} <= set(res["sim_results"])

    old = MLSimulator(predictor=AreaPredictor(), export="qgeometry", workdir=tmp_path / "old").predict(comp)
    assert old["export"]["convention"] == "qgeometry" and old["export"]["clearance_um"] > 0
    assert geom_hash(old["design"]["gds"]) != geom_hash(res["design"]["gds"])

    cross = sim.predict({"design": {"design_options": XMON, "component_class": "TransmonCross"}})
    assert cross["export"]["convention"] == "qgeometry"
    xmon = sim.predict({"design": {"design_options": XMON, "component_class": "Xmon"}})
    assert xmon["export"]["family"] == "xmon" and xmon["export"]["ground_gap_um"] > 0
    assert {"cross_to_claw", "cross_to_ground"} <= set(xmon["sim_results"])
    with pytest.raises(ValueError, match="dataset family"):
        MLSimulator(predictor=AreaPredictor(), export="dataset").predict(
            {"design": {"design_options": XMON, "component_class": "TransmonCross"}}
        )
    with pytest.raises(ValueError, match="export must be"):
        MLSimulator(export="gds")


def test_row_notes_set_the_star_style_and_flag_irreproducible_rows(sim):
    row = {
        "design": {"design_options": {"pad_enables": "1,0,0,0,0"}, "component_class": "GeneralizedCapStar"},
        "notes": {"design_id": "ref/exp11/q3d_cap_1330", "export_style": "reexport", "geometry_reproducible": False},
    }
    res = sim.predict(row)
    assert res["export"]["export_style"] == "reexport"
    assert any("q3d_cap_1330" in w for w in res["warnings"])
    flat = pd.Series(
        {
            "design_options": {"pad_enables": "1,0,0,0,0"},
            "coupler_type": "GeneralizedCapStar",
            "export_style": "original",
            "geometry_reproducible": True,
        }
    )
    assert sim.predict(flat)["export"]["export_style"] == "original"
    gap = sim.predict({"design": {"design_options": XMON, "component_class": "Xmon"}, "notes": {"ground_gap_um": 40.0}})
    assert gap["export"]["ground_gap_um"] == 40.0


def test_calcs_helpers_match_the_existing_transmon_cross_calcs():
    C = np.array([[50.0, -30.0, -20.0], [-30.0, 34.0, -4.0], [-20.0, -4.0, 24.0]])
    sr = {"cross_to_claw": 4.0, "cross_to_ground": 30.0}
    assert EC_MHz(sr) == pytest.approx(EC_MHz_from_matrix(C, ["G", "S", "N"], "S"), rel=1e-9)
    assert 400 < EC_MHz(sr) < 700
    g = g_MHz(sr, EJ_GHz=15.0, f_r_GHz=7.0, res_type="quarter")
    assert np.isfinite(g) and g > 0
    floating = EC_MHz_from_matrix(C, ["G", "S", "N"], ["S", "N"])
    assert floating > 0
    with pytest.raises(KeyError):
        transmon_cross_capacitances({"north_to_south": 1.0})


@pytest.mark.integration
def test_qcap_on_a_squadds_row(headless_qiskit_environment):
    """Networked (model download): one GeneralizedCapNInterdigital row through the released QCap model."""
    sim = MLSimulator(threads=int(os.environ.get("SQUADDS_TEST_THREADS", "2")))
    res = sim.predict({"design": {"design_options": {}, "component_class": "GeneralizedCapNInterdigital"}})
    sr = res["sim_results"]
    assert res["fidelity"] == "ml:QCap" and res["checks"]["passed"]
    assert 0 < sr["north_to_south"] < sr["north_to_north"]
