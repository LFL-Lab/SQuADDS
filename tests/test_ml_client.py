"""Client of the SQuADDS ML inference API (``squadds.ml.client``).

The unit tests replace ``requests.get`` / ``requests.post`` with fakes that record each call. The live tests (marker
``integration``) call the hosted API: listing, model contracts, a prediction with the contract's example request, and
QCap / QCap-Fast predictions of bundled layouts, checked against the local model when it can be loaded.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import requests

from squadds.ml import client

PUBLIC_MODEL = "transmon_cross_hamiltonian_inverse"


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


@pytest.fixture
def calls(monkeypatch):
    log = []

    def fake_get(url, headers=None, timeout=None):
        log.append(("GET", url, headers, timeout, None))
        if url.endswith("/v1/models"):
            return FakeResponse(body={"models": [{"id": PUBLIC_MODEL}]})
        return FakeResponse(body={"id": url.rsplit("/", 1)[-1]})

    def fake_post(url, json=None, files=None, data=None, headers=None, timeout=None):
        log.append(("POST", url, headers, timeout, {"json": json, "files": files, "data": data}))
        return FakeResponse(body={"outputs": {}, "metadata": {}, "timing_s": {}, "cached": False})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.delenv("SQUADDS_ML_API_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return log


def test_list_models_default_url_without_token(calls):
    assert client.list_models() == [{"id": PUBLIC_MODEL}]
    method, url, headers, _, _ = calls[0]
    assert (method, url, headers) == ("GET", f"{client.DEFAULT_URL}/v1/models", {})


def test_base_url_from_env_and_argument(calls, monkeypatch):
    monkeypatch.setenv("SQUADDS_ML_API_URL", "https://example.test/")
    client.get_model("m1")
    client.get_model("m2", base_url="https://other.test")
    assert calls[0][1] == "https://example.test/v1/models/m1"
    assert calls[1][1] == "https://other.test/v1/models/m2"


def test_token_from_argument_or_env(calls, monkeypatch):
    client.list_models(token="tok-arg")
    monkeypatch.setenv("HF_TOKEN", "tok-env")
    client.list_models()
    assert calls[0][2] == {"Authorization": "Bearer tok-arg"}
    assert calls[1][2] == {"Authorization": "Bearer tok-env"}


def test_predict_json_payload(calls):
    client.predict(PUBLIC_MODEL, inputs={"qubit_frequency_GHz": 4.85}, options={"include_scaled_outputs": True})
    method, url, _, timeout, payload = calls[0]
    assert (method, url) == ("POST", f"{client.DEFAULT_URL}/v1/models/{PUBLIC_MODEL}/predict")
    assert payload["json"] == {"inputs": {"qubit_frequency_GHz": 4.85}, "options": {"include_scaled_outputs": True}}
    assert payload["files"] is None and timeout == 600


def test_predict_file_from_path_and_bytes(calls, tmp_path):
    gds = tmp_path / "chip.gds"
    gds.write_bytes(b"GDSDATA")
    client.predict("file_model", file=gds, inputs={"ground_layer": "1/0"})
    client.predict("file_model", file=b"RAW")
    first, second = calls[0][4], calls[1][4]
    assert first["files"] == {"file": ("layout.gds", b"GDSDATA")}
    assert json.loads(first["data"]["inputs"]) == {"ground_layer": "1/0"}
    assert json.loads(first["data"]["options"]) == {}
    assert second["files"] == {"file": ("layout.gds", b"RAW")}


@pytest.mark.parametrize(
    ("status", "body", "text", "expected"),
    [
        (404, {"error": {"code": "model_not_found", "message": "no such model"}}, "", "404"),
        (422, {"detail": "bad inputs"}, "", "bad inputs"),
        (502, None, "upstream down", "upstream down"),
    ],
)
def test_errors_raise_with_status_and_message(monkeypatch, status, body, text, expected):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(status, body, text))
    with pytest.raises(RuntimeError, match=expected) as e:
        client.get_model("x")
    assert str(status) in str(e.value)


def _wait_until_up(deadline_s=240):
    """A sleeping Space answers 503 while it boots; keep asking until it lists models."""
    t0 = time.monotonic()
    while True:
        try:
            return client.list_models(timeout=60)
        except (RuntimeError, requests.ConnectionError, requests.Timeout) as e:
            if time.monotonic() - t0 > deadline_s or ("503" not in str(e) and isinstance(e, RuntimeError)):
                raise
            time.sleep(10)


@pytest.mark.integration
def test_live_hosted_api_lists_describes_and_predicts():
    models = {m["id"]: m for m in _wait_until_up()}
    assert PUBLIC_MODEL in models and models[PUBLIC_MODEL]["status"] == "ready"

    contract = client.get_model(PUBLIC_MODEL)
    assert contract["input_order"] == ["qubit_frequency_GHz", "anharmonicity_MHz"]
    request = contract["example"]["request"]

    envelope = client.predict(PUBLIC_MODEL, inputs=request["inputs"], options=request["options"], timeout=300)
    assert envelope["model_id"] == PUBLIC_MODEL and envelope["revision"] == contract["revision"]
    (row,) = envelope["outputs"]["predictions"]
    assert list(row) == contract["output_order"]
    for name, value in row.items():
        assert contract["output_units"][name] == "m"
        assert math.isfinite(value) and 1e-7 < value < 1e-2, (name, value)


QCAP = {"qcap": "QCap", "qcap-fast": "QCap-Fast"}
CAPML = Path(__file__).parent / "data" / "capml" / "parity"
LAYOUTS = ("exp13_q3d_cap_0414.gds", "random_v1_q3d_cap_0000.gds")  # first sent as a path, second as bytes


@pytest.fixture(scope="module")
def live_qcap():
    """The hosted QCap and QCap-Fast predictions of the bundled layouts, one request each."""
    _wait_until_up()
    return {
        (mid, f): client.predict(mid, file=CAPML / f if i == 0 else (CAPML / f).read_bytes(), timeout=600)
        for mid in QCAP
        for i, f in enumerate(LAYOUTS)
    }


@pytest.mark.integration
@pytest.mark.parametrize("model_id", list(QCAP))
def test_live_hosted_qcap_lists_describes_and_predicts(live_qcap, model_id):
    models = {m["id"]: m for m in client.list_models(timeout=60)}
    assert models[model_id]["status"] == "ready"

    contract = client.get_model(model_id)
    assert contract["display_name"] == contract["model"]["name"] == QCAP[model_id]
    assert contract["input_kind"] == "file" and contract["units"] == "fF"

    for f in LAYOUTS:
        envelope = live_qcap[(model_id, f)]
        assert envelope["model_id"] == model_id and envelope["revision"] == contract["revision"], f
        assert envelope["metadata"]["weights_sha256"] == contract["model"]["weights_sha256"], f
        names, C = envelope["outputs"]["names"], np.array(envelope["outputs"]["matrix_fF"])
        assert names[0] == "G" and all(n.startswith("C") for n in names[1:]), (f, names)
        assert C.shape == (len(names), len(names)), f
        np.testing.assert_allclose(C, C.T, rtol=0, atol=1e-9)
        off = C[~np.eye(len(names), dtype=bool)]
        assert np.all(off < 0) and np.all(np.diag(C) > 0), (f, C)
        assert envelope["metadata"]["checks"]["passed"], (f, envelope["metadata"]["checks"])


@pytest.mark.integration
@pytest.mark.parametrize("model_id", list(QCAP))
def test_live_hosted_qcap_matches_local_model(live_qcap, released_capml_model, model_id):
    """The Space (x86, 2 threads) and this runner agree to 5e-3 relative on every matrix entry."""
    from squadds.ml.capacitance import CapacitancePredictor

    p = CapacitancePredictor(released_capml_model(QCAP[model_id]))
    for f in LAYOUTS:
        out = live_qcap[(model_id, f)]["outputs"]
        res = p.predict(str(CAPML / f))
        assert res.names == out["names"], f
        A = np.array(out["matrix_fF"])
        rel = float(np.max(np.abs(res.matrix_fF - A) / np.maximum(np.abs(A), 1e-12)))
        assert rel <= 5e-3, (f, rel)
