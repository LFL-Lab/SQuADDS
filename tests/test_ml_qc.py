"""Label / layout QC of planar capacitance data (squadds.ml.capacitance.qc). No network, no [ml] model."""

import numpy as np
import pandas as pd
import pytest

from squadds.ml.capacitance import qc as Q

gdstk = pytest.importorskip("gdstk")
pytest.importorskip("shapely")


def maxwell(x=10.0, y=8.0, z=2.0):
    """G, C1, C2 with closure exact: ground-C1 x, ground-C2 y, C1-C2 z (fF)."""
    return np.array([[x + y, -x, -y], [-x, x + z, -z], [-y, -z, y + z]])


def write_gds(path, n_pads=2, touching=False, unit=1e-6):
    lib = gdstk.Library(unit=unit, precision=1e-9)
    cell = lib.new_cell("main")
    s = 1e-6 / unit
    holes = [gdstk.rectangle((s * (40 + 120 * k), s * 40), (s * (140 + 120 * k), s * 140)) for k in range(n_pads)]
    ground = gdstk.boolean(
        gdstk.rectangle((0, 0), (s * (200 + 120 * n_pads), s * 200)), holes, "not", layer=1, datatype=0
    )
    cell.add(*ground)
    for k in range(n_pads):
        x0 = s * (60 + 120 * k)
        cell.add(gdstk.rectangle((x0, s * 60), (x0 + s * 60, s * 120), layer=1, datatype=10))
        if touching and k == 0:
            cell.add(gdstk.rectangle((x0 + s * 50, s * 60), (x0 + s * 70, s * 80), layer=1, datatype=10))
    lib.write_gds(str(path))
    return path


def test_clean_matrix_passes():
    r = Q.label_qc(maxwell(), ["G", "C1", "C2"])
    assert r.ok, r.errors
    assert r.info["closure_max_rel"] < 1e-12
    assert r.info["signal_block_min_eig_fF"] > 0
    assert "PASS" in r.to_markdown()


def test_units_and_reorder():
    C = maxwell()
    r = Q.label_qc(C[np.ix_([1, 0, 2], [1, 0, 2])] * 1e-3, ["C1", "G", "C2"], net_order=["G", "C1", "C2"], unit="pF")
    assert r.ok, r.errors
    np.testing.assert_allclose(r.info["matrix_fF"], C)
    assert np.allclose(Q.to_fF(1e15, "pF*1e15"), 1e3)
    assert np.allclose(Q.to_fF(2e-15, "F"), 2.0)
    with pytest.raises(ValueError):
        Q.to_fF(1.0, "mF")


def test_unit_mistake_is_warned():
    r = Q.label_qc(maxwell() * 1e-15, ["G", "C1", "C2"], unit="fF")
    assert any("farads" in w for w in r.warnings)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda C: C.__setitem__((0, 1), C[0, 1] * 1.2), "asymmetric"),
        (lambda C: (C.__setitem__((1, 2), 0.5), C.__setitem__((2, 1), 0.5)), "positive off-diagonal"),
        (lambda C: C.__setitem__((1, 1), -1.0), "non-positive diagonal"),
        (lambda C: (C.__setitem__((1, 2), 0.0), C.__setitem__((2, 1), 0.0)), "exactly-zero"),
    ],
)
def test_rejections(mutate, needle):
    C = maxwell()
    mutate(C)
    r = Q.label_qc(C, ["G", "C1", "C2"])
    assert not r.ok
    assert any(needle in e for e in r.errors), r.errors


def test_small_asymmetry_warns_and_symmetrizes():
    C = maxwell()
    C[0, 1] *= 1 + 1e-5
    r = Q.label_qc(C, ["G", "C1", "C2"])
    assert r.ok, r.errors
    assert any("slightly asymmetric" in w for w in r.warnings)
    np.testing.assert_allclose(r.info["matrix_fF"], r.info["matrix_fF"].T)


def test_flagged_zero_allowed_and_pd():
    C = maxwell()
    C[1, 2] = C[2, 1] = 0.0
    assert Q.label_qc(C, ["G", "C1", "C2"], flagged_zero=[("C2", "C1")]).ok
    bad = np.array([[3.0, -1, -1], [-1, 1.0, -2.0], [-1, -2.0, 1.0]])
    r = Q.label_qc(bad, ["G", "C1", "C2"])
    assert any("positive definite" in e for e in r.errors)


def test_net_names():
    assert not Q.label_qc(maxwell(), ["N", "C1", "C2"]).ok
    assert not Q.label_qc(maxwell(), ["G", "C1", "C1"]).ok
    assert not Q.label_qc(maxwell(), ["G", "C1"]).ok
    r = Q.label_qc(maxwell(), ["G", "C1", "C2"], net_order=["C1", "G", "C2"])
    assert any("must be first" in e for e in r.errors)


def test_closure_warning():
    C = maxwell()
    C[1, 1] *= 1.2
    r = Q.label_qc(C, ["G", "C1", "C2"])
    assert r.ok
    assert any("closure" in w for w in r.warnings)


def test_pass_history_and_convergence(tmp_path):
    C = maxwell() * 1e-3
    hdr = ["Freq [GHz]", "Pass []"] + [f"C({a},{b}) [pF]" for a in ("C1", "G", "C2") for b in ("C1", "G", "C2")]
    order = [1, 0, 2]
    rows = []
    for k, s in enumerate((0.98, 0.995, 1.0)):
        M = C * s
        rows.append([13, k + 1] + [M[i, j] for i in order for j in order])
    rows.append([13, 4] + [""] * 9)
    p = tmp_path / "cap_matrix.csv"
    import csv

    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([hdr, *rows])
    m = Q.read_cap_matrix(p)
    assert m["names"] == ["G", "C1", "C2"]
    label = m["passes"][2][1]
    np.testing.assert_allclose(label, maxwell())
    conv = {"delta_pct_final": 0.08, "delta_target_pct": 0.05, "passes_completed": 3}
    r = Q.label_qc(label, m["names"], passes=m["passes"], convergence=conv)
    assert r.ok, r.errors
    assert r.info["converged"] is False and any("not converged" in w for w in r.warnings)
    assert 0.4 < r.info["last_pass_change_max_pct"] < 0.6
    r2 = Q.label_qc(m["passes"][1][1], m["names"], passes=m["passes"])
    assert any("last complete pass" in e for e in r2.errors)
    from squadds.ml.capacitance.training_data import read_q3d_csv

    names, last, _ = read_q3d_csv(p)
    np.testing.assert_allclose(last, label)


def test_read_conv(tmp_path):
    p = tmp_path / "x.conv"
    p.write_text(
        "Completed : 3\nMaximum   : 40\nTarget    : 0.05\nCurrent   : 0.048\n"
        "Pass|# Triangle| Delta %|\n   1|  100|     N/A|\n   2|  200|  1.3|\n   3|  300|0.048|\n",
        encoding="utf-8",
    )
    c = Q.read_conv(p)
    assert c["passes_completed"] == 3 and c["delta_pct_final"] == 0.048 and len(c["pass_history"]) == 3


def test_gds_qc_counts_conductors(tmp_path):
    g = Q.gds_qc(write_gds(tmp_path / "a.gds"))
    assert g.ok, g.errors
    assert g.info["n_conductors"] == 3
    assert g.info["geom_hash"].startswith("sha256:")
    assert Q.conductor_count_qc(3, ["G", "C1", "C2"]).ok
    assert not Q.conductor_count_qc(3, ["G", "C1"]).ok
    t = Q.gds_qc(write_gds(tmp_path / "b.gds", touching=True))
    assert t.info["n_conductors"] == 3 and t.info["touching_polygons_joined"] == 1
    n = Q.gds_qc(write_gds(tmp_path / "c.gds"), nets={"C1": [0, 1]})
    assert n.info["n_conductors"] == 2


def test_geom_hash_unit_invariant(tmp_path):
    a = Q.geom_hash(write_gds(tmp_path / "um.gds", unit=1e-6))
    b = Q.geom_hash(write_gds(tmp_path / "nm.gds", unit=1e-9))
    assert a == b


def test_gds_qc_errors(tmp_path):
    bad = tmp_path / "bad.gds"
    bad.write_bytes(b"not a gds")
    assert not Q.gds_qc(bad).ok
    r = Q.gds_qc(write_gds(tmp_path / "a.gds"), layer_map={"ground": [[1, 0]], "conductor": [[2, 0]]})
    assert any("have no polygons" in e for e in r.errors)


def test_duplicates():
    ref = pd.DataFrame(
        {"design_id": ["ref/x/1", "ref/x/2"], "geom_hash": ["sha256:a", "sha256:b"], "split": ["train", "test"]}
    )
    r = Q.duplicate_qc({"d1": "sha256:a", "d2": "sha256:c", "d3": "sha256:c"}, ref)
    assert r.ok and len(r.warnings) == 2
    r = Q.duplicate_qc({"d1": "sha256:b"}, ref)
    assert not r.ok and "held-out test" in r.errors[0]
    assert any("not checked" in w for w in Q.duplicate_qc({"d": "x"}, None).warnings)


def test_stack():
    r = Q.stack_qc({"eps_r": 11.45, "substrate_um": 350}, 13)
    assert r.ok and r.info["training_stack"] and not r.warnings
    r = Q.stack_qc({"eps_r": 9.8, "substrate_um": 500}, 5)
    assert r.ok and any("training stack" in w for w in r.warnings)
    assert any("outside" in w for w in Q.stack_qc({"eps_r": 50, "substrate_um": 350}).warnings)
    assert not Q.stack_qc({"eps_r": 11.45}).ok


def test_prior_sanity_with_given_prior():
    class P:
        C_fF = maxwell()
        prior_version = "test"
        timing_s = {"total_s": 0.1}
        machine = "m"
        threads = 2

    C = maxwell()
    r = Q.prior_sanity(C, ["G", "C1", "C2"], None, prior=P())
    assert r.ok and not r.warnings and r.info["prior_median_rel_diff_pct"] == 0
    C2 = C.copy()
    C2[1, 2] = C2[2, 1] = -10.0
    r = Q.prior_sanity(C2, ["G", "C1", "C2"], None, prior=P())
    assert r.ok and any("outside" in w for w in r.warnings)
