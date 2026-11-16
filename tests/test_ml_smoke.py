"""Cross-platform smoke test of the released planar capacitance models, downloaded from the Hugging Face Hub.

Predicts a bundled held-out Xmon layout with ``load_model("QCap")`` and ``load_model("QCap-Fast")``. Needs the ``[ml]`` extra
and read access to the model repos: ``HF_TOKEN`` is read from the environment when set. When the repo cannot be read (401,
403 or 404) the test skips with the repo and the HTTP status in the reason."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from squadds import ml as ML

DATA = Path(__file__).parent / "data" / "capml"
GDS = DATA / "xmon_exp13_0032.gds"
REF = json.loads((DATA / "xmon_exp13_0032_stack.json").read_text(encoding="utf-8"))
STORED = {"QCap": "flagship/accurate", "QCap-Fast": "flagship/fast"}  # reference keys at storage time
TOL = 5e-3  # same cross-platform bound as the parity test (priors differ by up to ~2.4e-3 between macOS and x86)

pytestmark = pytest.mark.skipif(bool(ML.missing_ml_dependencies()), reason="squadds[ml] extra not installed")


def model_repo(name):
    from squadds.ml.capacitance.models import MANIFEST, model_key

    return MANIFEST["models"][model_key(name)]["repo_id"]


def _load_from_hub(name, monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    from squadds.ml.capacitance import load_model
    from squadds.ml.capacitance.models import ENV_MODELS

    monkeypatch.delenv(ENV_MODELS, raising=False)
    try:
        return load_model(name)
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403, 404):
            who = "HF_TOKEN" if os.environ.get("HF_TOKEN") else "no token (HF_TOKEN not set)"
            pytest.skip(f"{who} cannot read {model_repo(name)} ({status})")
        raise


@pytest.mark.parametrize("name", ["QCap", "QCap-Fast"])
def test_released_model_predicts_bundled_gds(name, monkeypatch):
    from squadds.ml.capacitance import CapacitancePredictor

    model = _load_from_hub(name, monkeypatch)
    r = CapacitancePredictor(model).predict(str(GDS))
    n = len(REF["names"])
    assert r.names == REF["names"]
    assert r.matrix_fF.shape == (n, n)
    assert np.all(np.isfinite(r.matrix_fF))
    assert r.checks["passed"], r.checks
    ref = np.array([float.fromhex(v) for v in REF["training_stack_reference"][STORED[name]]]).reshape(n, n)
    rel = float(np.max(np.abs(r.matrix_fF - ref) / np.maximum(np.abs(ref), 1e-12)))
    assert rel <= TOL, (name, rel)
