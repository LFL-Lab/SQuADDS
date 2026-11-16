"""MCP planar capacitance workflow tools: design (GDS, Hamiltonian), training / scoring / label QC, contributions, SQuADDS_DB
sources, the guide resources and the workflow prompts.

No model download and no network: predictions use stub predictors, scores a stub report, contributions stub the uploader.
"""

import asyncio
import csv
import json
import py_compile
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from squadds_mcp.prompts import capacitance_workflows as PW
from squadds_mcp.resources import planar_capacitance_guide as G
from squadds_mcp.tools import capacitance_contribution as K
from squadds_mcp.tools import capacitance_design as D
from squadds_mcp.tools import capacitance_training as T
from squadds_mcp.tools import database as DB

ROOT = Path(__file__).resolve().parents[1]
DESIGN_TOOLS = {
    "get_capacitance_component_info",
    "generate_component_gds",
    "predict_capacitance_from_design",
    "capacitance_to_hamiltonian",
    "solve_coupled_transmon_hamiltonian",
    "generate_two_transmon_gds",
    "correct_half_wave_resonator",
}
TRAINING_TOOLS = {
    "check_capacitance_label",
    "build_capacitance_training_shards",
    "plan_capacitance_training",
    "run_capacitance_fit_smoke",
    "score_capacitance_model",
}
CONTRIB_TOOLS = {
    "init_capacitance_contribution",
    "contribute_capacitance_data",
    "validate_capacitance_model",
    "contribute_capacitance_model",
}
OTHER_TOOLS = {
    "compute_capacitance_prior",
    "inspect_capacitance_layout",
    "get_planar_capacitance_design",
    "nearest_planar_capacitance_designs",
    "predict_closest_designs_ml",
    "list_squadds_db_sources",
    "enable_squadds_db_source",
    "disable_squadds_db_source",
}
PROMPTS = {
    "gds_to_hamiltonian",
    "design_planar_capacitor_to_target",
    "retrain_capacitance_model",
    "contribute_capacitance_data_or_model",
    "design_coupled_transmons",
}
RESOURCES = {"squadds://planar-capacitance-guide", "squadds://planar-capacitance-tutorials"}


def server(*registers):
    mcp = FastMCP(name="test")
    for r in registers:
        r(mcp)
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


@pytest.fixture(scope="module")
def main_server():
    from squadds_mcp.server import create_server

    return create_server()


@pytest.fixture
def no_ml(monkeypatch):
    import squadds.ml as M

    monkeypatch.setattr(M, "missing_ml_dependencies", lambda: ["torch"])


# ------------------------------------------------------------------------------------------------ registration
def test_registered_in_main_server(main_server):
    tools = {t.name for t in asyncio.run(main_server.list_tools())}
    assert DESIGN_TOOLS | TRAINING_TOOLS | CONTRIB_TOOLS | OTHER_TOOLS <= tools
    assert PROMPTS <= {p.name for p in asyncio.run(main_server.list_prompts())}
    assert RESOURCES <= {str(r.uri) for r in asyncio.run(main_server.list_resources())}


def test_tools_have_agent_docs(main_server):
    for t in asyncio.run(main_server.list_tools()):
        if t.name in DESIGN_TOOLS | TRAINING_TOOLS | CONTRIB_TOOLS | OTHER_TOOLS:
            assert t.description and len(t.description) > 80, t.name


def test_startup_sources(monkeypatch):
    from squadds_mcp.server import startup_sources

    monkeypatch.delenv("SQUADDS_MCP_SOURCES", raising=False)
    assert startup_sources() is None
    monkeypatch.setenv("SQUADDS_MCP_SOURCES", "default, planar_capacitance")
    assert startup_sources() == ("default", "planar_capacitance")


# ------------------------------------------------------------------------------------------------ Hamiltonian
def test_hamiltonian_grounded():
    C = [[200, -150, -30], [-150, 170, -10], [-30, -10, 45]]
    h = D.hamiltonian_from_maxwell(C, ["G", "cross", "claw"], qubit_type="xmon")
    from scipy.constants import Planck, e

    assert h["mode"] == "grounded" and h["C_sigma_fF"] == 170 and h["C_g_fF"] == 10
    assert h["EC_MHz"] == pytest.approx(e**2 / (2 * 170e-15) / Planck * 1e-6)
    assert "hint" in h and "qubit_frequency_GHz" not in h
    h = D.hamiltonian_from_maxwell(
        C, ["G", "cross", "claw"], qubit_type="xmon", EJ_GHz=15.0, resonator_frequency_GHz=7.0
    )
    assert 3 < h["qubit_frequency_GHz"] < 8 and h["anharmonicity_MHz"] < 0 and h["g_MHz"] > 0
    assert h["Lj_nH"] == pytest.approx(10.897, rel=1e-3)


def test_hamiltonian_target_ej_matches_analyzer_rule():
    from scqubits.core.transmon import Transmon

    C = [[200, -150, -30], [-150, 170, -10], [-30, -10, 45]]
    h = D.hamiltonian_from_maxwell(
        C, ["G", "cross", "claw"], qubit_type="xmon", qubit_frequency_GHz=5, anharmonicity_MHz=-200
    )
    assert h["EJ_GHz"] == pytest.approx(Transmon.find_EJ_EC(5, -0.2)[0])
    assert "target" in h["EJ_source"]


def test_hamiltonian_floating_pair():
    C = np.array([[300, -100, -90, -10], [-100, 120, -15, -2], [-90, -15, 110, -3], [-10, -2, -3, 20.0]])
    names = ["G", "pad_top", "pad_bot", "readout_pad"]
    h = D.hamiltonian_from_maxwell(C, names, qubit_type="transmon_pocket")
    inv = np.linalg.inv(C[1:3, 1:3])
    assert h["mode"] == "floating"
    assert h["C_sigma_fF"] == pytest.approx(1 / (inv[0, 0] + inv[1, 1] - 2 * inv[0, 1]))
    x, y = 120 - 15, 110 - 15
    assert h["C_g_fF"] == pytest.approx(abs((y * 2 - x * 3) / (x + y)))


def test_hamiltonian_errors_and_missing_preset_coupler():
    C = [[200, -150, -30], [-150, 170, -10], [-30, -10, 45]]
    assert D.hamiltonian_from_maxwell([[200, -150], [-150, 170]], ["G", "cross"], qubit_type="xmon")["coupler"] is None
    with pytest.raises(ValueError, match="ground"):
        D.hamiltonian_from_maxwell(C, ["G", "C1", "C2"], qubit_nodes=["G"])
    with pytest.raises(ValueError, match="not in names"):
        D.hamiltonian_from_maxwell(C, ["G", "C1", "C2"], qubit_nodes=["cross"])
    with pytest.raises(ValueError, match="qubit_type"):
        D.hamiltonian_from_maxwell(C, ["G", "C1", "C2"], qubit_type="fluxonium")
    with pytest.raises(ValueError, match="square"):
        D.hamiltonian_from_maxwell(C, ["G", "C1"], qubit_nodes=["C1"])


def test_hamiltonian_tool(monkeypatch):
    mcp = server(D.register_capacitance_design_tools)
    C = [[200, -150, -30], [-150, 170, -10], [-30, -10, 45]]
    out = call(mcp, "capacitance_to_hamiltonian", {"matrix_fF": C, "names": ["G", "C1", "C2"], "qubit_nodes": ["C1"]})
    assert out["C_sigma_fF"] == 170 and out["coupler"] is None
    pred = {"names": ["G", "C1", "C2"], "matrix_fF": C, "model": "QCap", "timing_s": {}}
    monkeypatch.setattr(D, "predict_gds", lambda *a, **k: {**pred, "threads": 2, "machine": "m"})
    out = call(mcp, "capacitance_to_hamiltonian", {"gds_path": "x.gds", "qubit_nodes": ["C1"], "coupler": "C2"})
    assert out["C_g_fF"] == 10 and out["prediction"]["model"] == "QCap"
    assert "error" in call(mcp, "capacitance_to_hamiltonian", {})


# ------------------------------------------------------------------------------------------------ components and GDS
def test_component_info():
    out = D.component_info()
    fams = {r["family"]: r for r in out["families"]}
    assert {"interdigital", "concentric", "star", "xmon", "pocket", "shielded_concentric"} <= set(fams)
    assert fams["xmon"]["squadds_db_config"] == "qubit-Xmon-cap_matrix"
    assert fams["xmon"]["net_names"]["S"] == "cross"
    assert "error" in D.component_info("TransmonCross")
    x = D.component_info("xmon")
    assert x["component_class"] == "Xmon" and "cross_length" in x["default_options"]


def test_generate_gds(tmp_path):
    pytest.importorskip("gdstk")
    out = D.generate_gds("Xmon", {"cross_length": "180um"}, str(tmp_path / "x" / "xmon.gds"))
    assert out.get("error") is None, out
    assert Path(out["gds_path"]).is_file() and out["layout"]["n_conductors"] == 3
    assert out["export"]["family"] == "xmon" and out["design_options"]["cross_length"] == "180um"
    assert "error" in D.generate_gds("Xmon", None, str(tmp_path / "x.txt"))
    assert "error" in D.generate_gds("NotAComponent", None, str(tmp_path / "y.gds"))


def test_predict_design_with_stub(monkeypatch, tmp_path):
    pytest.importorskip("gdstk")
    from squadds.ml.capacitance.result import CapacitanceResult

    C = np.array([[180.0, -40, -90], [-40, 42, -1.5], [-90, -1.5, 95]])

    class Stub:
        def predict(self, lay, **kw):
            return CapacitanceResult(["G", "C1", "C2"], C, "QCap", "accurate", timing_s={"total_s": 0.1}, machine="m")

    monkeypatch.setattr(D, "_ml_missing", lambda: None)
    monkeypatch.setattr(D, "_predictor", lambda model, threads: Stub())
    mcp = server(D.register_capacitance_design_tools)
    out = call(mcp, "predict_capacitance_from_design", {"component_class": "Xmon", "save_gds_dir": str(tmp_path)})
    assert out.get("error") is None, out
    assert out["names"][0] == "ground" and set(out["names"][1:]) == {"cross", "claw"}
    assert out["fidelity"] == "ml:QCap" and Path(out["gds_path"]).is_file()
    h = D.hamiltonian(out["matrix_fF"], out["names"], qubit_type="xmon")
    assert h["C_sigma_fF"] in (42.0, 95.0) and h["C_g_fF"] == 1.5


def test_design_tools_degrade_without_ml(no_ml, tmp_path):
    for out in (
        D.generate_gds("Xmon", None, str(tmp_path / "a.gds")),
        D.predict_design("Xmon", None),
        T.build_training_shards([{"gds": "a"}], str(tmp_path / "s")),
        T.fit_smoke(str(tmp_path), str(tmp_path / "r")),
        T.score_model(data=str(tmp_path)),
        K.validate_model_repo("me/model", local_dir=str(tmp_path)),
    ):
        assert "squadds[ml]" in out["install"] and "torch" in out["error"]
    from squadds_mcp.tools import capacitance_ml as M

    gds = tmp_path / "a.gds"
    gds.write_bytes(b"x")
    assert "install" in M.physics_prior(str(gds))


# ------------------------------------------------------------------------------------------------ label QC
def maxwell():
    return np.array([[50.0, -20, -10], [-20, 30, -5], [-10, -5, 20]])


def test_check_label_matrix():
    out = T.check_label(matrix_fF=maxwell().tolist(), names=["G", "C1", "C2"])
    assert out["ok"] and out["matrix_fF"][0][0] == 50
    bad = maxwell()
    bad[0, 1] = bad[1, 0] = 2
    out = T.check_label(matrix_fF=bad.tolist(), names=["G", "C1", "C2"])
    assert not out["ok"] and any("positive off-diagonal" in e for e in out["errors"])
    out = T.check_label(matrix_fF=(maxwell() * 1e-3).tolist(), names=["G", "C1", "C2"], unit="pF")
    assert out["ok"] and out["matrix_fF"][1][1] == pytest.approx(30)
    assert "error" in T.check_label()
    assert "not found" in T.check_label(path="nope.csv")["error"]


def test_check_label_q3d_csv(tmp_path):
    C = maxwell() * 1e-3
    hdr = ["Freq [GHz]", "Pass []"] + [f"C({a},{b}) [pF]" for a in ("C1", "G", "C2") for b in ("C1", "G", "C2")]
    order = [1, 0, 2]
    rows = [[13, k + 1] + [(C * s)[i, j] for i in order for j in order] for k, s in enumerate((0.99, 1.0))]
    rows.append([13, 3] + [""] * 9)
    p = tmp_path / "cap_matrix.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([hdr, *rows])
    out = call(server(T.register_capacitance_training_tools), "check_capacitance_label", {"path": str(p)})
    assert out["ok"], out["errors"]
    assert out["passes"] == 2 and out["names"] == ["G", "C1", "C2"]
    np.testing.assert_allclose(out["matrix_fF"], maxwell())


# ------------------------------------------------------------------------------------------------ training
def test_build_shards_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_ml_missing", lambda: None)
    out = T.build_training_shards(
        [{"gds": f"{k}.gds"} for k in range(T.MAX_INPROCESS_DESIGNS + 1)], str(tmp_path / "s")
    )
    assert "plan_capacitance_training" in out["hint"]
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x", encoding="utf-8")
    assert "not empty" in T.build_training_shards([{"gds": "a.gds"}], str(tmp_path / "full"))["error"]
    assert "no designs" in T.build_training_shards([], str(tmp_path / "s"))["error"]


def test_plan_fit_slurm(tmp_path):
    mcp = server(T.register_capacitance_training_tools)
    out = call(
        mcp,
        "plan_capacitance_training",
        {
            "mode": "fit",
            "out_dir": str(tmp_path / "runs"),
            "script_dir": str(tmp_path / "scripts"),
            "shard_dir": str(tmp_path / "shards"),
            "seeds": [0, 1],
            "partition": "gpu",
            "account": "lab",
            "setup": "module load python",
            "python": "python3",
        },
    )
    job = Path(out["files"]["slurm"]).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in job and "#SBATCH --partition=gpu" in job and "#SBATCH --account=lab" in job
    assert "module load python" in job and "--steps 18000" in job
    assert "--seeds 0,1" in out["run_locally"] and out["run_on_slurm"].startswith("sbatch ")


@pytest.mark.parametrize("mode", ["build", "in_family", "collective"])
def test_plan_driver_modes(tmp_path, mode):
    designs = tmp_path / "designs.json"
    designs.write_text(json.dumps([{"gds": "a.gds", "labels": "a.csv"}]), encoding="utf-8")
    kw = {"designs_json": str(designs), "shard_dir": str(tmp_path / "shards")}
    if mode == "collective":
        kw["base_shards"] = str(tmp_path / "base")
    out = T.plan_training(mode, str(tmp_path / "runs"), str(tmp_path / "scripts"), **kw)
    assert out.get("error") is None, out
    driver = Path(out["files"]["driver"])
    py_compile.compile(str(driver), doraise=True)
    text = driver.read_text(encoding="utf-8")
    assert (
        "'None'" not in text and repr(str(designs)) in text
    )  # embedded as a Python literal (escaped backslashes on Windows)
    assert not (tmp_path / "runs").exists()
    if mode != "build":
        assert "must not exist" in out["note"]


def test_plan_errors(tmp_path):
    p = lambda mode, **kw: T.plan_training(mode, str(tmp_path / "r"), str(tmp_path / "s"), **kw)  # noqa: E731
    assert "mode must be" in p("train")["error"]
    assert "shard_dir" in p("fit")["error"]
    assert "designs_json" in p("build", shard_dir="x")["error"]
    assert "not found" in p("in_family", designs_json=str(tmp_path / "nope.json"))["error"]
    assert "base_shards" in p("collective", shard_dir="x")["error"]


def test_fit_smoke_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_ml_missing", lambda: None)
    out = T.fit_smoke(str(tmp_path), str(tmp_path / "r"), steps=T.MAX_SMOKE_STEPS + 1)
    assert "capped" in out["error"] and "plan_capacitance_training" in out["hint"]
    assert "not found" in T.fit_smoke(str(tmp_path / "nope"), str(tmp_path / "r"))["error"]


def test_score_guards_and_stub(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_ml_missing", lambda: None)
    assert "maintainer-only" in T.score_model(data=str(tmp_path), split="test")["error"]
    assert "capped" in T.score_model(layouts=[{"gds": "a", "labels": "b"}] * (T.MAX_SCORE_LAYOUTS + 1))["error"]
    assert "error" in T.score_model()
    from squadds.ml.capacitance import benchmark as B

    overall = pd.DataFrame([{"kind": "mutual", "layouts": 3, "median_pct": 0.3, "mae_fF": 0.05, "rmse_fF": 0.07}])
    seen = {}

    def fake_score(model, data, **kw):
        seen.update(kw, model=model, data=data)
        return B.ScoreReport(overall, overall, overall, {"seconds": 1.0, "threads": 2, "machine": "m"}, "val", "QCap")

    monkeypatch.setattr(B, "score", fake_score)
    out = call(server(T.register_capacitance_training_tools), "score_capacitance_model", {"data": str(tmp_path)})
    assert out["headline"] == {"kind": "mutual", "layouts": 3, "median_pct": 0.3, "mae_fF": 0.05, "rmse_fF": 0.07}
    assert "MAE" in out["summary"] and "RMSE" in out["summary"] and out["timing"]["machine"] == "m"
    assert seen["split"] == "val" and seen["model"] == "QCap"


# ------------------------------------------------------------------------------------------------ contributions
def test_init_contribution(tmp_path):
    out = K.init_contribution(str(tmp_path / "c"))
    assert {"manifest.yaml", "designs.csv", "README.md"} <= set(out["files"])
    assert (tmp_path / "c" / "gds").is_dir() and "Maxwell" in out["readme"]
    assert "not empty" in K.init_contribution(str(tmp_path / "c"))["error"]


def test_contribute_data_is_dry_run_by_default(monkeypatch, tmp_path):
    from squadds.ml.capacitance import contribute as C

    calls = []

    def fake(path, **kw):
        calls.append(kw)
        return {
            "ok": True,
            "folder": "contrib/x-1",
            "files": ["a"],
            "report": "r",
            "pr_url": None if kw["dry_run"] else "u",
        }

    monkeypatch.setattr(C, "contribute_capacitance", fake)
    mcp = server(K.register_capacitance_contribution_tools)
    out = call(mcp, "contribute_capacitance_data", {"path": str(tmp_path)})
    assert out["dry_run"] is True and "confirm=True" in out["note"] and calls[-1]["dry_run"] is True
    out = call(mcp, "contribute_capacitance_data", {"path": str(tmp_path), "dry_run": False})
    assert "confirm=True" in out["error"] and len(calls) == 1
    out = call(mcp, "contribute_capacitance_data", {"path": str(tmp_path), "dry_run": False, "confirm": True})
    assert out["pr_url"] == "u" and calls[-1]["dry_run"] is False
    assert "not found" in K.contribute_data(str(tmp_path / "nope"))["error"]


def test_contribute_data_refuses_invalid(tmp_path):
    out = K.contribute_data(str(tmp_path), reference="none")
    assert out["ok"] is False and "failed validation" in out["error"]


def test_contribute_model_confirm(monkeypatch, tmp_path):
    from squadds.ml.capacitance import contribute as C

    calls = []

    def fake(repo, revision, **kw):
        calls.append(kw)
        return {"ok": True, "entry": {"id": "x"}, "report": "r", "files": ["model_zoo.yaml"], "pr_url": None}

    monkeypatch.setattr(K, "_ml_missing", lambda: None)
    monkeypatch.setattr(C, "contribute_model", fake)
    out = K.contribute_model_repo("me/model", local_dir=str(tmp_path))
    assert out["dry_run"] is True and calls[-1]["dry_run"] is True and calls[-1]["local_dir"] == str(tmp_path)
    assert "confirm=True" in K.contribute_model_repo("me/model", dry_run=False)["error"]
    assert len(calls) == 1


# ------------------------------------------------------------------------------------------------ SQuADDS_DB sources + search
class FakeDB:
    def __init__(self):
        self._default_configs = ["qubit-TransmonCross-cap_matrix"]
        self.configs = list(self._default_configs)
        self._extra_sources = {}
        self.calls = []

    def enabled_sources(self):
        return ["default", *self._extra_sources]

    def enable_source(self, name, **options):
        if name != "planar_capacitance":
            raise ValueError(f"Unknown SQuADDS_DB source {name!r}")
        self.calls.append(options)

        class Src:
            def configs(self):
                return ["qubit-Xmon-cap_matrix"]

        self._extra_sources[name] = Src()
        self.configs += ["qubit-Xmon-cap_matrix"]

    def unselect_all(self):
        self.calls.append("unselect")

    def select_system(self, s):
        self.calls.append(("system", s))

    def select_coupler(self, c):
        self.calls.append(("coupler", c))


def test_sources_helpers():
    db = FakeDB()
    assert DB.sources_info(db)["enabled"] == ["default"]
    out = DB.enable_source(db, "planar_capacitance", revision="abc")
    assert out["added_configs"] == ["qubit-Xmon-cap_matrix"] and db.calls[-1] == {"revision": "abc"}
    info = DB.sources_info(db)
    assert info["configs_by_source"] == {"planar_capacitance": ["qubit-Xmon-cap_matrix"]} and info["total_configs"] == 2
    assert "planar_capacitance" in info["available"]
    assert "Unknown" in DB.enable_source(db, "nope")["error"]


def test_coupler_search_configuration():
    from squadds_mcp.tools.analysis import _configure_db_for_search

    db = FakeDB()
    _configure_db_for_search(db, "coupler", coupler="GeneralizedCapStar")
    assert db.calls == ["unselect", ("system", "coupler"), ("coupler", "GeneralizedCapStar")]


# ------------------------------------------------------------------------------------------------ resources and prompts
def _tool_names(main_server):
    return {t.name for t in asyncio.run(main_server.list_tools())}


def test_guide_numbers_match_docs():
    docs = (ROOT / "docs" / "source" / "ml_models.rst").read_text(encoding="utf-8")
    triples = re.findall(r"([\d.]+) %\s+\((?:MAE\s+)?([\d.]+) fF\s+/\s+(?:RMSE\s+)?([\d.]+) fF\)", G.GUIDE)
    assert len(triples) >= 12
    for pct, mae, rmse in triples:
        assert re.search(
            rf"{re.escape(pct)} %\s+\((?:MAE\s+)?{re.escape(mae)} fF\s+/\s+(?:RMSE\s+)?{re.escape(rmse)} fF\)", docs
        ), (
            pct,
            mae,
            rmse,
        )
    for t in ("0.845 / 5.96 s", "1.88 s", "23.0 s", "3.76 s", "46.2 s", "0.093 s", "0.17 s", "45,771", "35,354"):
        assert t in G.GUIDE and t in docs, t


def test_resources_read(main_server):
    guide = asyncio.run(main_server.read_resource("squadds://planar-capacitance-guide"))
    text = list(guide)[0].content
    assert "never merge nets" in text.lower() and "SQuADDS_PlanarCapacitance-heldout" in text
    assert "private until publication" not in text.lower()
    tut = list(asyncio.run(main_server.read_resource("squadds://planar-capacitance-tutorials")))[0].content
    for n in range(11, 17):
        assert f"Tutorial {n}:" in tut
    for t in G.TUTORIALS:
        assert (ROOT / "tutorials" / f"{t['file']}.ipynb").is_file(), t["file"]


def test_guide_and_tutorials_name_real_tools(main_server):
    tools = _tool_names(main_server)
    for t in G.TUTORIALS:
        assert set(t["tools"]) <= tools, t["tools"]
    named = set(re.findall(r"`([a-z_]+)\(", G.GUIDE))
    assert named and named <= tools, named - tools


@pytest.mark.parametrize("prompt", sorted(PROMPTS))
def test_prompts_render_and_name_real_tools(main_server, prompt):
    res = asyncio.run(main_server.get_prompt(prompt, {}))
    text = res.messages[0].content.text
    assert "MAE and RMSE" in text
    named = set(re.findall(r"`([a-z_]+)\(", text))
    assert named and named <= _tool_names(main_server), named - _tool_names(main_server)


def test_prompt_arguments():
    mcp = server(PW.register_capacitance_workflow_prompts)
    res = asyncio.run(mcp.get_prompt("gds_to_hamiltonian", {"gds_path": "my.gds", "qubit_type": "transmon_pocket"}))
    text = res.messages[0].content.text
    assert "my.gds" in text and 'qubit_type="transmon_pocket"' in text


# ------------------------------------------------------------------------------------------------ two coupled transmons
TT_TARGETS = {
    "f_q_A": 4.3,
    "eta_A": -0.214,
    "f_q_B": 4.6,
    "eta_B": -0.209,
    "f_r_A": 6.6,
    "kappa_A": 0.0016,
    "f_r_B": 6.5,
    "kappa_B": 0.0011,
    "J": 0.045,
    "chi_A": 0.0008,
    "chi_B": 0.0008,
}


@pytest.fixture(scope="module")
def tt_design():
    return D.coupled_transmon_design(TT_TARGETS)


def test_two_transmon_solution_matches_reference(tt_design):
    out = tt_design
    assert out.get("error") is None, out
    A, B = out["transmons"]["A"], out["transmons"]["B"]
    assert A["E_J (GHz)"] == pytest.approx(13.248, abs=1e-3) and B["E_J (GHz)"] == pytest.approx(15.258, abs=1e-3)
    assert A["E_C (MHz)"] == pytest.approx(191.09, abs=0.01) and B["E_C (MHz)"] == pytest.approx(188.50, abs=0.01)
    assert A["g = G n01 (MHz)"] == pytest.approx(123.06, abs=0.01) and B["g = G n01 (MHz)"] == pytest.approx(
        107.23, abs=0.01
    )
    assert A["L_J (nH)"] == pytest.approx(12.34, abs=0.01) and A["K (MHz)"] == pytest.approx(30.73, abs=0.01)
    assert max(abs(r["miss (%)"]) for r in out["hamiltonian_check"].values()) < 1e-6
    t = out["target_capacitances"]
    assert t["bodies"] == ["ground", "cross A", "cross B", "claw A", "claw B"]
    assert t["five_controlled_fF"]["C_sigma_A (cross A diagonal)"] == pytest.approx(101.628, abs=1e-3)
    assert max(abs(x) for x in t["solve_residual_pct"]) < 1e-6 and "ideal" in t["reference"]
    assert out["resonators"]["A"]["find_closest_designs"]["target_params"]["kappa_kHz"] == pytest.approx(1600)
    assert any("NOT a training family" in v for v in out["validity"])
    assert any("half the full dispersive shift" in v for v in out["conventions"])


def test_two_transmon_layout_evaluation(tt_design):
    M = tt_design["target_capacitances"]["matrix_fF"]
    out = D.coupled_transmon_design(TT_TARGETS, reference_matrix_fF=M, layout_matrix_fF=M)
    check = out["layout"]["hamiltonian_check"]
    assert out["layout"]["worst_miss_pct"] < 0.5
    assert max(abs(check[k]["miss (%)"]) for k in check if not k.startswith("chi")) < 1e-6
    assert all(abs(v["miss_fF"]) < 1e-9 for v in out["layout"]["capacitance_miss"].values())
    assert len(out["layout"]["line_frequency_GHz"]) == 2 and min(out["layout"]["node_capacitance_fF"]) > 0
    assert out["resonators"]["A"]["find_closest_designs"]["system_type"] == "qubit_cavity"
    assert "correct_half_wave_resonator" in out["resonators"]["A"]["correction"]
    assert out["target_capacitances"]["Z0_ohm"] == 50.0


def test_two_transmon_target_errors():
    out = D.coupled_transmon_design({"f_q_A": 4.3})
    assert "missing targets" in out["error"] and "chi_B" in out["error"]
    out = D.coupled_transmon_design(TT_TARGETS | {"eta_A": 0.2})
    assert "eta_A must be negative" in out["error"]
    assert "5 x 5" in D.coupled_transmon_design(TT_TARGETS, reference_matrix_fF=[[1.0]])["error"]


def test_two_transmon_gds(monkeypatch, tmp_path):
    pytest.importorskip("gdstk")
    pytest.importorskip("klayout.db")  # the Qiskit Metal to GDS export needs the gds extra
    from squadds.ml.capacitance.result import CapacitanceResult

    out = D.two_transmon_gds([170, 160, 0.8, 0.8, 20], "interdigital", str(tmp_path / "x" / "tt.gds"))
    assert out.get("error") is None, out
    assert Path(out["gds_path"]).is_file() and out["geometry_um"]["claw_A"] == pytest.approx(0.8 * 155)
    assert "IDC" in out["fixed_um"] and "warning" not in out
    assert "warning" in D.two_transmon_gds([170, 170, 0.8, 0.8, 20], "gap", str(tmp_path / "s.gds"))
    assert "variant" in D.two_transmon_gds([170, 160, 0.8, 0.8, 20], "bridge", str(tmp_path / "b.gds"))["error"]
    assert "p = " in D.two_transmon_gds([1, 2], "gap", str(tmp_path / "c.gds"))["error"]

    C = np.diag([300.0, 110, 112, 40, 41])
    for i, j, v in [(0, 1, 80), (0, 2, 82), (1, 2, 2), (1, 3, 12), (2, 4, 11), (0, 3, 25), (0, 4, 26)]:
        C[i, j] = C[j, i] = -v

    class Stub:
        def predict(self, lay, **kw):
            return CapacitanceResult(["G", "C1", "C2", "C3", "C4"], C, "QCap", "accurate", machine="m")

    monkeypatch.setattr(D, "_predictor", lambda model, threads: Stub())
    out = D.two_transmon_gds([170, 160, 0.8, 0.8, 20], "gap", str(tmp_path / "p.gds"), predict=True)
    M = np.asarray(out["matrix_fF"])
    assert M.shape == (5, 5) and out["model"] == "QCap" and np.allclose(M, M.T)
    assert sorted(np.diag(M)) == sorted(np.diag(C))


def test_two_transmon_gds_without_ml(no_ml, tmp_path):
    assert "install" in D.two_transmon_gds([170, 160, 0.8, 0.8, 20], "gap", str(tmp_path / "a.gds"))


def test_coupled_transmons_prompt_carries_caveats():
    mcp = server(PW.register_capacitance_workflow_prompts)
    text = asyncio.run(mcp.get_prompt("design_coupled_transmons", {"variant": "interdigital"})).messages[0].content.text
    for phrase in (
        "NOT a training family",
        "ground_spacing = 0",
        "11.7 um",
        "QCap-Fast only for step directions",
        "half the full dispersive shift",
        "dressed eta is not a well-posed target",
        "run_q3d.py",
        "L_A != L_B",
    ):
        assert phrase in text, phrase
    assert 'variant="interdigital"' in text and "15]" in text
