"""Parity of squadds.ml.capacitance with the tl-squadds Space code (integration: needs the released models).

Eleven public layouts of ``SQuADDS/SQuADDS_PlanarCapacitance`` (train/val splits, one per family plus a 12-conductor
layout) with the Space code's QCap and QCap-Fast matrices (stored under the Space's keys flagship / fast), stored on the
machine named in the reference. On that machine the match is within 1e-5 (library versions move the last digits: shapely
2.2.0 instead of 2.1.2 moves QCap-Fast by 2.6e-6 on one layout); elsewhere within 5e-3, because the physics priors differ
slightly between platforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

DATA = Path(__file__).parent / "data" / "capml"
REF = json.loads((DATA / "parity_reference.json").read_text(encoding="utf-8"))


STORED = {"QCap": "flagship", "QCap-Fast": "fast"}  # reference keys at storage time


@pytest.mark.parametrize("name", ["QCap", "QCap-Fast"])
def test_matches_space_code(released_capml_model, name):
    from squadds.ml.capacitance import CapacitancePredictor
    from squadds.ml.capacitance._runtime import machine

    tol = 1e-5 if machine() == REF["machine"] else 5e-3
    p = CapacitancePredictor(released_capml_model(name))
    for r in REF["records"]:
        res = p.predict(str(DATA / "parity" / r["file"]))
        assert res.names == r["names"], r["file"]
        A = np.array(r[STORED[name]])
        rel = float(np.max(np.abs(res.matrix_fF - A) / np.maximum(np.abs(A), 1e-12)))
        assert rel <= tol, (r["file"], rel)
        assert res.checks["passed"], (r["file"], res.checks)
