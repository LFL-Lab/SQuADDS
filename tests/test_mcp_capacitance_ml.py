"""MCP planar capacitance tools (squadds_mcp.tools.capacitance_ml): registration, degradation without [ml], payloads.

No model download and no network: predictions use a stub predictor, the dataset a local Parquet copy.
"""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools import capacitance_ml as T

TOOLS = {
    "list_capacitance_models",
    "predict_capacitance_from_gds",
    "query_planar_capacitance",
    "get_planar_capacitance_conventions",
    "validate_capacitance_contribution",
}


def server():
    mcp = FastMCP(name="test")
    T.register_capacitance_ml_tools(mcp)
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


def test_tools_registered():
    names = {t.name for t in asyncio.run(server().list_tools())}
    assert TOOLS <= names


def test_registered_in_main_server():
    from squadds_mcp.server import create_server

    names = {t.name for t in asyncio.run(create_server().list_tools())}
    assert TOOLS <= names


def test_conventions():
    c = call(server(), "get_planar_capacitance_conventions")
    assert c["units"]["capacitance"] == "fF"
    assert "never merged" in c["matrix"] and "G" in c["ground"]
    assert c["stack"]["eps_r"] == 11.45
    assert c["models"]["QCap"]["default"] is True and not c["models"]["QCap-Fast"]["default"]
    assert "QCap-Fast only" in c["models"]["QCap-Fast"]["warning"]
    assert set(c["model_guide"]) == {"QCap", "QCap-Fast"}
    assert "private_until_publication" not in c["dataset"]


def test_list_models():
    out = call(server(), "list_capacitance_models")
    rows = {r["model"]: r for r in out["models"]}
    assert set(rows) == {"QCap", "QCap-Fast"} and rows["QCap"]["default"]
    assert rows["QCap"]["repo_id"] == "SQuADDS/QCap"
    ref = rows["QCap"]["reference"]
    assert {"median_pct", "mae_ff", "rmse_ff"} <= set(ref["val"])


def test_degrades_without_ml(monkeypatch, tmp_path):
    import squadds.ml as M

    monkeypatch.setattr(M, "missing_ml_dependencies", lambda: ["torch"])
    gds = tmp_path / "a.gds"
    gds.write_bytes(b"x")
    out = T.predict_gds(str(gds))
    assert "squadds[ml]" in out["install"] and "torch" in out["error"]
    assert "note" in call(server(), "list_capacitance_models")
    assert "install" in T.validate_contribution(str(tmp_path), check_prior=True)


def test_predict_with_stub(monkeypatch, tmp_path):
    from squadds.ml.capacitance.result import CapacitanceResult

    C = np.array([[18.0, -10, -8], [-10, 12, -2], [-8, -2, 10]])
    res = CapacitanceResult(["G", "C1", "C2"], C, "QCap", "accurate", timing_s={"total_s": 0.5}, machine="m")
    calls = {}

    class Stub:
        def predict(self, path, **kw):
            calls.update(kw, path=path)
            return res

    monkeypatch.setattr(T, "_ml_missing", lambda: None)
    monkeypatch.setattr(T, "_predictor", lambda model, threads: Stub())
    gds = tmp_path / "chip.gds"
    gds.write_bytes(b"x")
    out = call(server(), "predict_capacitance_from_gds", {"gds_path": str(gds), "conductor_layers": "1/10,2/10"})
    assert out["names"] == ["G", "C1", "C2"] and out["matrix_fF"][0][1] == -10
    assert out["checks"]["passed"] and out["units"] == "fF"
    assert out["sim_results"]["C1_to_C2"] == 2.0
    assert calls["conductor_layers"] == [(1, 10), (2, 10)]
    assert "not found" in T.predict_gds(str(tmp_path / "missing.gds"))["error"]


def test_query_local_dataset(tmp_path):
    d = tmp_path / "data" / "designs"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "design_id": ["ref/exp6/a", "ref/exp10/b"],
            "family": ["interdigital", "concentric"],
            "paper_family": ["interdigital", "concentric"],
            "model_family": ["ncap", "concentric"],
            "split": ["train", "val"],
            "collection": ["exp6", "exp10"],
            "n_conductors": [3, 3],
            "squadds_config": ["coupler-GeneralizedCapNInterdigital-cap_matrix", None],
        }
    ).to_parquet(d / "designs-train.parquet")
    out = T.query_dataset(family="concentric", local_dir=str(tmp_path))
    assert out["total"] == 1 and out["rows"][0]["design_id"] == "ref/exp10/b"
    assert "planar_capacitance" in out["squadds_db"]
    assert "error" in T.query_dataset(local_dir=str(tmp_path / "nope"))


def test_validate_tool(tmp_path):
    out = call(server(), "validate_capacitance_contribution", {"path": str(tmp_path / "nope")})
    assert "not found" in out["error"]
    out = call(server(), "validate_capacitance_contribution", {"path": str(tmp_path), "reference": "none"})
    assert out["ok"] is False and any("manifest.yaml" in e for e in out["errors"])


@pytest.mark.parametrize("s, want", [("1/0", [(1, 0)]), ("1/10, 2/10", [(1, 10), (2, 10)])])
def test_layers(s, want):
    assert T._layers(s) == want


def test_conventions_are_json():
    json.dumps(T.conventions())
