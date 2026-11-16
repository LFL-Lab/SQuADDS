"""MCP hosted ML API tools (squadds_mcp.tools.hosted_ml): registration, listing, prediction, errors.

No network: ``requests.get`` / ``requests.post`` are replaced by fakes that record the calls.
"""

import asyncio
import json

import pytest
import requests
from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools import hosted_ml as H

TOOLS = {"list_hosted_ml_models", "predict_with_hosted_ml_model"}
MODEL = "transmon_cross_hamiltonian_inverse"
LISTING = {
    "models": [
        {"id": MODEL, "family": "keras_tabular", "input_kind": "json", "status": "ready", "revision": "a" * 40},
    ]
}
DETAIL = {**LISTING["models"][0], "input_order": ["qubit_frequency_GHz", "anharmonicity_MHz"]}
ENVELOPE = {
    "model_id": MODEL,
    "family": "keras_tabular",
    "revision": "a" * 40,
    "outputs": {"predictions": [{"design_options.cross_length": 0.0002}]},
    "metadata": {"output_units": {"design_options.cross_length": "m"}},
    "timing_s": {"total": 0.01, "queue": 0.0},
    "cached": False,
}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self.body, self.text = status_code, body, json.dumps(body)

    def json(self):
        return self.body


def server():
    mcp = FastMCP(name="test")
    H.register_hosted_ml_tools(mcp)
    return mcp


def call(mcp, name, args=None):
    out = asyncio.run(mcp.call_tool(name, args or {}))
    if isinstance(out, tuple):
        out = out[1]
    if isinstance(out, dict) and set(out) == {"result"}:
        out = out["result"]
    if isinstance(out, list):
        out = json.loads(out[0].text)
    return out


@pytest.fixture
def http(monkeypatch):
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append(("GET", url, None))
        if url.endswith("/v1/models"):
            return FakeResponse(200, LISTING)
        if url.endswith(f"/v1/models/{MODEL}"):
            return FakeResponse(200, DETAIL)
        return FakeResponse(404, {"error": {"code": "unknown_model", "message": "no such model"}})

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        seen.append(("POST", url, json))
        if f"/v1/models/{MODEL}/predict" in url:
            return FakeResponse(200, ENVELOPE)
        return FakeResponse(404, {"error": {"code": "unknown_model", "message": "no such model"}})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.delenv("SQUADDS_ML_API_URL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return seen


def test_tools_registered():
    assert TOOLS <= {t.name for t in asyncio.run(server().list_tools())}


def test_registered_in_main_server():
    from squadds_mcp.server import create_server

    assert TOOLS <= {t.name for t in asyncio.run(create_server().list_tools())}


def test_list_with_details(http):
    out = call(server(), "list_hosted_ml_models")
    assert out["api"] == "https://squadds-squadds-ml-inference-api.hf.space"
    assert out["models"] == [DETAIL]
    assert [u for _, u, _ in http] == [f"{out['api']}/v1/models", f"{out['api']}/v1/models/{MODEL}"]


def test_list_without_details_and_base_url(http):
    out = call(server(), "list_hosted_ml_models", {"base_url": "http://localhost:7860/", "details": False})
    assert out == {"api": "http://localhost:7860", "models": LISTING["models"]}
    assert [u for _, u, _ in http] == ["http://localhost:7860/v1/models"]


def test_env_base_url(http, monkeypatch):
    monkeypatch.setenv("SQUADDS_ML_API_URL", "http://mirror.example")
    out = call(server(), "list_hosted_ml_models", {"details": False})
    assert out["api"] == "http://mirror.example" and http[0][1] == "http://mirror.example/v1/models"


def test_predict_sends_inputs_and_options(http):
    inputs = {"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0}
    args = {"model_id": MODEL, "inputs": inputs, "options": {"include_scaled_outputs": False}}
    out = call(server(), "predict_with_hosted_ml_model", args)
    assert out == ENVELOPE
    method, url, body = http[0]
    assert method == "POST" and url.endswith(f"/v1/models/{MODEL}/predict")
    assert body == {"inputs": inputs, "options": {"include_scaled_outputs": False}}


def test_predict_batch(http):
    rows = [{"qubit_frequency_GHz": 4.85, "anharmonicity_MHz": -205.0}] * 2
    call(server(), "predict_with_hosted_ml_model", {"model_id": MODEL, "inputs": rows})
    assert http[0][2] == {"inputs": rows, "options": {}}


def test_unknown_model_is_an_error_not_an_exception(http):
    out = call(server(), "predict_with_hosted_ml_model", {"model_id": "nope", "inputs": {"x": 1}})
    assert out["model_id"] == "nope" and "404" in out["error"] and "unknown_model" in out["error"]


def test_unreachable_api_is_an_error(monkeypatch):
    def down(*a, **kw):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", down)
    out = call(server(), "list_hosted_ml_models")
    assert out["error"].startswith("ConnectionError") and "api" in out
