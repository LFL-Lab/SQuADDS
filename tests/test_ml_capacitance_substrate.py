"""Calibrated substrate route on a tested point of the stack study (integration: needs the released models).

Held-out Xmon exp13/q3d_cap_0032 of the 15-design stack DoE. Off the training stack QCap solves the prior at the
requested stack (recorded in ``result.substrate``). The training-stack output must be bit-identical to the code before the
calibrated route on the machine that stored the reference (skipped elsewhere)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

DATA = Path(__file__).parent / "data" / "capml"
GDS = DATA / "xmon_exp13_0032.gds"
REF = json.loads((DATA / "xmon_exp13_0032_stack.json").read_text(encoding="utf-8"))
S, P1, G = 2, 3, 0  # C1 = N (readout claw), C2 = S (cross), C3 = P1 (bus claw), file order


@pytest.fixture(scope="module")
def models(released_capml_model):
    from squadds.ml.capacitance import CapacitancePredictor

    return {k: CapacitancePredictor(released_capml_model(k)) for k in ("QCap", "QCap-Fast")}


def test_qcap_off_stack_solves_the_prior_at_the_stack(models):
    for eps_r, t_um in ((11.45, 10.0), (3.8, 350.0)):
        r = models["QCap"].predict(str(GDS), eps_r=eps_r, thickness_um=t_um)
        assert r.names == REF["names"] and r.checks["passed"]
        assert r.substrate["prior_solved_at"] == {"eps_r": eps_r, "thickness_um": t_um} and r.substrate["tested"]


def test_training_stack_bit_identical(models):
    from squadds.ml.capacitance._runtime import machine

    ref = REF["training_stack_reference"]
    if machine() != ref["machine"]:
        pytest.skip(f"reference stored on {ref['machine']}")
    for k, p in models.items():
        r = p.predict(str(GDS))
        stored = {"QCap": "flagship/accurate", "QCap-Fast": "flagship/fast"}[k]  # keys at storage time
        assert [float(v).hex() for v in r.matrix_fF.ravel()] == ref[stored], k
        assert r.substrate["route"] == "training stack"


def test_qcap_fast_off_stack_is_rescaled_and_warned(models):
    p = models["QCap-Fast"]
    base = p.predict(str(GDS))
    with pytest.warns(UserWarning, match="QCap-Fast does not solve the prior at your stack"):
        r = p.predict(str(GDS), eps_r=9.8, thickness_um=200)
    np.testing.assert_allclose(r.matrix_fF, base.matrix_fF * (1 + 9.8) / (1 + 11.45), rtol=1e-12)
    assert r.substrate["route"].startswith("fast: eps-rescaled") and r.substrate["prior_solved_at"] is None


def test_one_layer_file_is_bitwise_the_two_layer_file(models, tmp_path):
    """Ground (1/0) and conductors (1/10) moved to one layer: the detected split and every model's matrix are bit-identical."""
    import gdstk

    from squadds.ml.capacitance.predictor import clear_cache

    lib = gdstk.read_gds(str(GDS))
    for c in lib.cells:
        for q in c.polygons:
            if (q.layer, q.datatype) in [(1, 0), (1, 10)]:
                q.layer, q.datatype = 9, 0
    one = tmp_path / "one_layer.gds"
    lib.write_gds(str(one), max_points=8190)
    for k, m in models.items():
        clear_cache()
        a = m.predict(str(GDS))
        clear_cache()
        with pytest.warns(UserWarning, match="detected automatically"):
            b = m.predict(str(one), ground_layers=[(9, 0)], conductor_layers=[(9, 0)])
        assert a.layout_hash == b.layout_hash and a.matrix_fF.tobytes() == b.matrix_fF.tobytes(), k
        assert b.ground_detection["confidence"] == "high"
