"""Xmon E_C from C_sigma including every claw; the TransmonCross convention is left alone."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from squadds.calcs.floating_transmon import XmonHamiltonian
from squadds.calcs.transmon_cross import TransmonCrossHamiltonian
from squadds.core.analysis import Analyzer
from squadds.ml.capacitance.bridge import EC_MHz_from_matrix

TARGET = {"qubit_frequency_GHz": 4.5, "anharmonicity_MHz": -200.0}


def xmon_row(c_g, c_claw, c_bus=None):
    return {
        "cross_to_cross": c_g + c_claw + (c_bus or 0.0),
        "cross_to_claw": c_claw,
        "cross_to_bus_claw": c_bus,
        "cross_to_ground": c_g,
        "claw_to_claw": 60.0 + c_claw,
        "claw_to_ground": 60.0,
        "bus_claw_to_ground": 50.0 if c_bus is not None else None,
    }


def analysis_for(rows, qubit="Xmon"):
    db = SimpleNamespace(selected_qubit=qubit, selected_cavity=None)
    a = SimpleNamespace(db=db, df=rows, H_param_keys=list(TARGET), target_params=TARGET, selected_resonator_type=None)
    return a


def test_two_claw_hand_example():
    # C_sigma = 150 + 4 + 3 = 157 fF -> E_C = e^2 / (2 h C_sigma) = 19.370229 GHz fF / 157 fF = 123.377 MHz
    H = XmonHamiltonian(analysis_for(pd.DataFrame([xmon_row(150.0, 4.0, 3.0)])))
    H.add_qubit_H_params()
    assert H.df["C_sigma_fF"].iloc[0] == pytest.approx(157.0, rel=1e-12)
    assert H.df["EC"].iloc[0] * 1e3 == pytest.approx(123.377, abs=1e-3)
    names = ["ground", "cross", "claw", "bus_claw"]
    C = [[0, 0, 0, 0], [-150.0, 157.0, -4.0, -3.0], [-60.0, -4.0, 64.0, 0.0], [-50.0, -3.0, 0.0, 53.0]]
    assert H.df["EC"].iloc[0] * 1e3 == pytest.approx(EC_MHz_from_matrix(C, names, "cross"), rel=1e-9)
    legacy = TransmonCrossHamiltonian.EC(H, cross_to_claw=4.0, cross_to_ground=150.0)
    assert legacy * 1e3 == pytest.approx(19370.229 / 154.0, rel=1e-6) and legacy > H.df["EC"].iloc[0]


def test_one_claw_matches_the_transmon_cross_convention():
    H = XmonHamiltonian(analysis_for(pd.DataFrame([xmon_row(120.0, 5.1)])))
    H.add_qubit_H_params()
    legacy = TransmonCrossHamiltonian.EC(H, cross_to_claw=5.1, cross_to_ground=120.0)
    assert H.df["EC"].iloc[0] == pytest.approx(legacy, rel=1e-12)


def test_missing_diagonal_uses_every_claw():
    row = xmon_row(150.0, 4.0, 3.0) | {"cross_to_cross": None}
    H = XmonHamiltonian(analysis_for(pd.DataFrame([row])))
    assert H.capacitances(pd.Series(row)) == pytest.approx((157.0, 4.0), rel=1e-12)


def test_analyzer_dispatches_xmon_rows():
    rows = pd.DataFrame([xmon_row(150.0, 4.0, 3.0), xmon_row(100.0, 6.0), xmon_row(80.0, 7.0, 2.0)])
    rows["design_options"] = [{"cross_length": f"{v}um"} for v in (300, 250, 200)]
    db = SimpleNamespace(
        selected_component_name="Xmon",
        selected_component="qubit",
        selected_data_type="cap_matrix",
        selected_confg="qubit-Xmon-cap_matrix",
        selected_qubit="Xmon",
        selected_cavity=None,
        selected_resonator_type=None,
        selected_coupler=None,
        selected_system="qubit",
        selected_df=rows,
        qubit_df=rows,
        cavity_df=None,
        coupler_df=None,
    )
    analyzer = Analyzer(db)
    analyzer.find_closest(TARGET, num_top=1, display=False)
    assert analyzer.df["C_sigma_fF"].tolist() == pytest.approx([157.0, 106.0, 89.0], rel=1e-12)
