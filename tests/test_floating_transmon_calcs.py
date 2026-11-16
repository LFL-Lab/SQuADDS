"""Floating-transmon Hamiltonian from the Maxwell matrix and the Analyzer's dispatch by qubit class."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from squadds.calcs.floating_transmon import (
    FloatingTransmonHamiltonian,
    XmonHamiltonian,
    floating_transmon_capacitances,
    maxwell_from_row,
    qubit_hamiltonian_class,
)
from squadds.calcs.transmon_cross import TransmonCrossHamiltonian, get_transmon_E01_alpha
from squadds.core.analysis import Analyzer

TARGET = {"qubit_frequency_GHz": 4.5, "anharmonicity_MHz": -200.0}
EC_GHZ_FF = 19.370229  # e^2 / (2 h) in GHz fF


def stub_db(qubit, df, system="qubit"):
    return SimpleNamespace(
        selected_component_name=qubit,
        selected_component="qubit",
        selected_data_type="cap_matrix",
        selected_confg=f"qubit-{qubit}-cap_matrix",
        selected_qubit=qubit,
        selected_cavity=None,
        selected_resonator_type=None,
        selected_coupler=None,
        selected_system=system,
        selected_df=df,
        qubit_df=df,
        cavity_df=None,
        coupler_df=None,
    )


def pocket_row(c12, c1g, c2g, c1r, c2r, c1b=None, c2b=None, cr_g=40.0, cb_g=30.0):
    """Magnitudes of a pocket row (closure diagonal), bus pad optional."""
    row = {
        "pad_top_to_pad_bot": c12,
        "pad_top_to_readout_pad": c1r,
        "pad_bot_to_readout_pad": c2r,
        "pad_top_to_ground": c1g,
        "pad_bot_to_ground": c2g,
        "readout_pad_to_ground": cr_g,
        "pad_top_to_bus_pad": c1b,
        "pad_bot_to_bus_pad": c2b,
        "bus_pad_to_ground": cb_g if c1b is not None else None,
    }
    row["pad_top_to_pad_top"] = c12 + c1g + c1r + (c1b or 0.0)
    row["pad_bot_to_pad_bot"] = c12 + c2g + c2r + (c2b or 0.0)
    row["readout_pad_to_readout_pad"] = cr_g + c1r + c2r
    row["bus_pad_to_bus_pad"] = cb_g + c1b + c2b if c1b is not None else None
    return row


def test_effective_capacitance_hand_example():
    row = pocket_row(c12=20.0, c1g=60.0, c2g=40.0, c1r=6.0, c2r=2.0)
    C, names = maxwell_from_row(row, ["pad_top", "pad_bot", "readout_pad", "bus_pad"])
    assert names == ["pad_top", "pad_bot", "readout_pad"]
    c_eff, c_g = floating_transmon_capacitances(C, names, coupler="readout_pad")
    # C1S = 66, C2S = 42: C_eff = 20 + 66*42/108 ; C_g = (42*6 - 66*2)/108
    assert c_eff == pytest.approx(20.0 + 66.0 * 42.0 / 108.0, rel=1e-12)
    assert c_g == pytest.approx((42.0 * 6.0 - 66.0 * 2.0) / 108.0, rel=1e-12)
    inv = np.linalg.inv(C[:2, :2])
    assert c_eff == pytest.approx(1.0 / (inv[0, 0] + inv[1, 1] - 2 * inv[0, 1]), rel=1e-12)


def test_coupling_capacitance_is_the_charge_free_voltage_division():
    row = pocket_row(c12=11.0, c1g=55.0, c2g=48.0, c1r=7.5, c2r=1.25, c1b=0.5, c2b=3.0)
    C, names = maxwell_from_row(row, ["pad_top", "pad_bot", "readout_pad", "bus_pad"])
    for coupler in ("readout_pad", "bus_pad"):
        c_eff, c_g = floating_transmon_capacitances(C, names, coupler=coupler)
        k = names.index(coupler)
        v_pads = -np.linalg.solve(C[:2, :2], C[:2, k])  # both pads charge free, coupler at 1 V, the rest grounded
        assert abs(v_pads[0] - v_pads[1]) * c_eff == pytest.approx(c_g, rel=1e-12)


def test_missing_diagonal_uses_closure():
    row = pocket_row(c12=20.0, c1g=60.0, c2g=40.0, c1r=6.0, c2r=2.0)
    full, _ = maxwell_from_row(row, ["pad_top", "pad_bot", "readout_pad"])
    for k in ("pad_top_to_pad_top", "pad_bot_to_pad_bot", "readout_pad_to_readout_pad"):
        row[k] = float("nan")
    closed, _ = maxwell_from_row(row, ["pad_top", "pad_bot", "readout_pad"])
    np.testing.assert_allclose(closed, full, rtol=1e-12)


def test_floating_hamiltonian_columns_follow_the_transmon_approximations():
    rows = pd.DataFrame(
        [
            pocket_row(c12=20.0, c1g=60.0, c2g=40.0, c1r=6.0, c2r=2.0),
            pocket_row(c12=11.0, c1g=55.0, c2g=48.0, c1r=7.5, c2r=1.25, c1b=0.5, c2b=3.0),
        ]
    )
    analysis = SimpleNamespace(
        db=stub_db("TransmonPocket", rows), df=rows, H_param_keys=list(TARGET), target_params=TARGET
    )
    analysis.selected_resonator_type = None
    H = FloatingTransmonHamiltonian(analysis)
    H.add_qubit_H_params()
    df = H.df
    EJ = TransmonCrossHamiltonian.EJ(H, TARGET["qubit_frequency_GHz"], TARGET["anharmonicity_MHz"] * 1e-3)
    for _, r in df.iterrows():
        C, names = maxwell_from_row(r, ["pad_top", "pad_bot", "readout_pad", "bus_pad"])
        c_eff, _ = floating_transmon_capacitances(C, names)
        assert r["C_sigma_fF"] == pytest.approx(c_eff, rel=1e-12)
        assert r["EC"] == pytest.approx(EC_GHZ_FF / c_eff, rel=1e-5)
        f01, alpha = get_transmon_E01_alpha(EJ, r["EC"])
        assert (r["qubit_frequency_GHz"], r["anharmonicity_MHz"]) == pytest.approx((f01, alpha), rel=1e-6)


def test_pocket_g_uses_the_effective_coupling_capacitance():
    rows = pd.DataFrame([pocket_row(c12=20.0, c1g=60.0, c2g=40.0, c1r=6.0, c2r=2.0)])
    rows["cavity_frequency_GHz"] = 7.0
    rows["resonator_type"] = "quarter"
    analysis = SimpleNamespace(
        db=stub_db("TransmonPocket", rows), df=rows, H_param_keys=list(TARGET), target_params=TARGET
    )
    analysis.selected_resonator_type = "quarter"
    H = FloatingTransmonHamiltonian(analysis)
    H.add_cavity_coupled_H_params()
    c_eff, c_g = 20.0 + 66.0 * 42.0 / 108.0, (42.0 * 6.0 - 66.0 * 2.0) / 108.0
    expected = H.g_from_cap_matrix(C=c_eff - c_g, C_c=c_g, EJ=H.df["EJ"].iloc[0], f_r=7.0, res_type="quarter")
    assert H.df["g_MHz"].iloc[0] == pytest.approx(expected, rel=1e-12) and expected > 0


def test_dispatch_by_qubit_class():
    assert qubit_hamiltonian_class("TransmonPocket") is FloatingTransmonHamiltonian
    assert qubit_hamiltonian_class("Xmon") is XmonHamiltonian
    for name in ("TransmonCross", None, ["qubit"]):
        assert qubit_hamiltonian_class(name) is None


def test_pocket_find_closest_through_the_analyzer():
    rows = pd.DataFrame(
        [
            pocket_row(c12=20.0, c1g=60.0, c2g=40.0, c1r=6.0, c2r=2.0),
            pocket_row(c12=11.0, c1g=55.0, c2g=48.0, c1r=7.5, c2r=1.25, c1b=0.5, c2b=3.0),
            pocket_row(c12=8.0, c1g=45.0, c2g=40.0, c1r=4.0, c2r=1.0),
        ]
    )
    rows["design_options"] = [{"pad_width": f"{w}um"} for w in (400, 450, 500)]
    analyzer = Analyzer(stub_db("TransmonPocket", rows))
    closest = analyzer.find_closest(TARGET, num_top=1, display=False)
    assert len(closest) == 1 and "EC" in analyzer.df and analyzer.df["C_sigma_fF"].notna().all()
    assert closest.index[0] in analyzer.df.index and analyzer.df["qubit_frequency_GHz"].between(2, 9).all()


# TransmonCross rows and their outputs at 32ecc3c (before the dispatch): the default path must not move
CROSS_ROWS = {"cross_to_ground": [158.4, 120.0, 95.5], "cross_to_claw": [3.7, 5.1, 8.25]}
CROSS_GOLDEN = {
    "EC": [0.1194955541311482, 0.15483796422589227, 0.1867010055388831],
    "EJ": [15.198655439117937] * 3,
    "qubit_frequency_GHz": [3.6881575676732403, 4.178006342702211, 4.569639973163476],
    "anharmonicity_MHz": [-129.37826225382486, -169.7918933178624, -206.9316690238061],
}


@pytest.mark.parametrize("qubit", ["TransmonCross", None])
def test_transmon_cross_outputs_are_unchanged(qubit):
    rows = pd.DataFrame({**CROSS_ROWS, "claw_to_ground": [90.9, 80.0, 70.0]})
    rows["design_options"] = [{"cross_length": f"{v}um"} for v in (300, 250, 200)]
    analyzer = Analyzer(stub_db(qubit, rows.copy()))
    assert type(analyzer._qubit_hamiltonian()) is TransmonCrossHamiltonian
    closest = analyzer.find_closest(TARGET, num_top=3, display=False)
    assert closest.index.tolist() == [2, 1, 0]
    assert "C_sigma_fF" not in analyzer.df
    for key, values in CROSS_GOLDEN.items():
        # EJ comes from scqubits' find_EJ_EC (BFGS on a squared cost, default gtol): its end point moves with the BLAS /
        # numpy build (7.4e-5 relative on Linux x86 against these macOS values), and f_q and alpha follow EJ. The
        # bit-exact check against TransmonCrossHamiltonian on this machine below guards the dispatch.
        rtol = 1e-12 if key == "EC" else 1e-3
        np.testing.assert_allclose(analyzer.df[key].to_numpy(), values, rtol=rtol, atol=0)

    reference = SimpleNamespace(db=stub_db(qubit, rows.copy()), H_param_keys=list(TARGET), target_params=TARGET)
    reference.df, reference.selected_resonator_type = reference.db.selected_df, None
    H = TransmonCrossHamiltonian(reference)
    H.add_qubit_H_params()
    pd.testing.assert_frame_equal(analyzer.df[H.df.columns], H.df, check_exact=True)
