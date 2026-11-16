"""MCP planar capacitance tools against the current SQuADDS API: every keyword the MCP passes binds to the real signature,
every tool parameter is used, and every accuracy % in the MCP text carries MAE and RMSE.

No model download and no network: library calls are replaced by recorders that bind to the real signatures.
"""

import ast
import asyncio
import inspect
import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from squadds_mcp.tools import analysis as AN
from squadds_mcp.tools import capacitance_contribution as K
from squadds_mcp.tools import capacitance_design as D
from squadds_mcp.tools import capacitance_ml as M
from squadds_mcp.tools import capacitance_training as T

ROOT = Path(__file__).resolve().parents[1]
GDS = ROOT / "tests" / "data" / "capml" / "xmon_exp13_0032.gds"
MAXWELL = [[50.0, -20.0, -10.0], [-20.0, 30.0, -5.0], [-10.0, -5.0, 20.0]]


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


def bind(fn, *args, **kwargs):
    inspect.signature(fn).bind(*args, **kwargs)


def named(fn) -> set[str]:
    kinds = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    return {k for k, p in inspect.signature(fn).parameters.items() if p.kind not in kinds}


def bind_predict(layout, kw):
    """CapacitancePredictor.predict keywords; the rest go through prepare_layout to the layout reader."""
    from squadds.ml.capacitance._core.layout import prepare
    from squadds.ml.capacitance.predictor import CapacitancePredictor

    own = named(CapacitancePredictor.predict)
    bind(CapacitancePredictor.predict, None, layout, **{k: v for k, v in kw.items() if k in own})
    bind(prepare, str(layout), **{k: v for k, v in kw.items() if k not in own})


def result(names, C):
    from squadds.ml.capacitance.result import CapacitanceResult

    return CapacitanceResult(list(names), np.asarray(C, float), "QCap", "accurate", machine="m")


class Recorder:
    def __init__(self, names, C):
        self.names, self.C, self.kw = names, C, {}

    def predict(self, layout, **kw):
        bind_predict(layout, kw)
        self.kw.update(kw)
        return result(self.names, self.C)


# ------------------------------------------------------------------------------------------------ keywords bind
def test_predict_gds_keywords_bind(monkeypatch):
    rec = Recorder(["G", "C1", "C2"], MAXWELL)
    monkeypatch.setattr(M, "_ml_missing", lambda: None)
    monkeypatch.setattr(M, "_predictor", lambda model, threads: rec)
    out = M.predict_gds(
        str(GDS), eps_r=9.8, thickness_um=500.0, ground_layers="2/0", conductor_layers="2/0", ground_pieces="1"
    )
    assert out.get("error") is None, out
    assert rec.kw["eps_r"] == 9.8 and rec.kw["thickness_um"] == 500.0 and rec.kw["ground_pieces"] == [1]


def test_hamiltonian_from_gds_forwards_every_option(monkeypatch):
    seen = {}

    def fake_predict_gds(*a, **kw):
        bind(M.predict_gds, *a, **kw)
        seen.update(kw)
        keys = ("model", "checks", "warnings", "substrate", "timing_s", "threads", "machine")
        return {"names": ["G", "cross", "claw"], "matrix_fF": MAXWELL, **{k: kw.get(k, "x") for k in keys}}

    monkeypatch.setattr(D, "predict_gds", fake_predict_gds)
    args = {"gds_path": "x.gds", "qubit_nodes": ["cross"], "coupler": "claw", "model": "QCap-Fast", "eps_r": 10.34}
    args |= {"thickness_um": 430.0, "ground_layers": "2/0", "conductor_layers": "2/0", "ground_pieces": "2"}
    out = call(server(D.register_capacitance_design_tools), "capacitance_to_hamiltonian", args)
    assert out.get("error") is None, out
    assert {k: seen[k] for k in ("model", "eps_r", "thickness_um", "ground_pieces")} == {
        "model": "QCap-Fast",
        "eps_r": 10.34,
        "thickness_um": 430.0,
        "ground_pieces": "2",
    }
    assert out["C_sigma_fF"] == 30.0 and "substrate" in out["prediction"]


def test_design_tools_pass_the_substrate(monkeypatch, tmp_path):
    pytest.importorskip("gdstk")
    pytest.importorskip("klayout.db")  # the Qiskit Metal to GDS export needs the gds extra
    rec = Recorder(["G", "C1", "C2"], [[180.0, -40, -90], [-40, 42, -1.5], [-90, -1.5, 95]])
    monkeypatch.setattr(D, "_ml_missing", lambda: None)
    monkeypatch.setattr(D, "_predictor", lambda model, threads: rec)
    args = {"component_class": "Xmon", "save_gds_dir": str(tmp_path), "eps_r": 10.34, "thickness_um": 430.0}
    out = call(server(D.register_capacitance_design_tools), "predict_capacitance_from_design", args)
    assert out.get("error") is None, out
    assert rec.kw["eps_r"] == 10.34 and rec.kw["thickness_um"] == 430.0 and "substrate" in out

    C = np.diag([300.0, 110, 112, 40, 41])
    for i, j, v in [(0, 1, 80), (0, 2, 82), (1, 2, 2), (1, 3, 12), (2, 4, 11), (0, 3, 25), (0, 4, 26)]:
        C[i, j] = C[j, i] = -v
    rec5 = Recorder(["G", "C1", "C2", "C3", "C4"], C)
    monkeypatch.setattr(D, "_predictor", lambda model, threads: rec5)
    out = D.two_transmon_gds([170, 160, 0.8, 0.8, 20], "gap", str(tmp_path / "t.gds"), predict=True, eps_r=9.8)
    assert out.get("error") is None, out
    assert rec5.kw["eps_r"] == 9.8 and out["ground_area_check"]["ok"]
    assert out["ground_area_check"]["in_file_um2"] == pytest.approx(
        out["ground_area_check"]["read_by_model_um2"], rel=1e-4
    )


def test_physics_prior_returns_warnings_ground_and_solves(monkeypatch):
    import squadds.ml.capacitance as CM
    from squadds.ml.capacitance import priors
    from squadds.ml.capacitance._core.layout import prepare

    lay = SimpleNamespace(
        warnings=["ground and conductors share a layer: ground = piece #1", "x is beyond the validated range"],
        info={"ground_detection": {"method": "automatic", "ground_pieces": [1]}},
    )

    def fake_prepare(layout, **kw):
        bind(prepare, layout, **kw)
        return lay

    def fake_prior(layout, **kw):
        bind(priors.optimized_physics_prior, layout, **kw)
        assert layout is lay
        warnings.warn("substrate outside the tested range", UserWarning, stacklevel=2)
        return priors.PriorResult(
            ["G", "C1"], np.array([[2.0, -1], [-1, 2]]), solves_fF={1.5: np.eye(2), 3.0: np.eye(2)}
        )

    monkeypatch.setattr(M, "_ml_missing", lambda: None)
    monkeypatch.setattr(CM, "prepare_layout", fake_prepare, raising=False)
    monkeypatch.setattr(CM, "optimized_physics_prior", fake_prior, raising=False)
    out = M.physics_prior(str(GDS), ground_layers="2/0", conductor_layers="2/0", eps_r=3.8, thickness_um=50.0)
    assert out.get("error") is None, out
    assert out["ground_detection"]["ground_pieces"] == [1] and set(out["solves_fF"]) == {"1.5", "3"}
    assert "substrate outside the tested range" in out["warnings"]
    assert any("share a layer" in w for w in out["warnings"])
    assert not any("validated range" in w for w in out["warnings"])


# ------------------------------------------------------------------------------------------------ layout inspection
def test_inspect_layout_two_layers_and_one_layer(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    out = M.inspect_gds(str(GDS), png_path=str(tmp_path / "xmon.png"))
    assert out.get("error") is None, out
    assert out["names"][0] == "G" and out["n_conductors"] == len(out["names"]) >= 2 and not out["single_layer"]
    assert out["ground_area_in_file_um2"] == pytest.approx(out["ground_area_um2"], rel=1e-4)
    assert Path(out["png_path"]).stat().st_size > 0

    lib = gdstk.read_gds(str(GDS))
    for cell in lib.cells:
        for poly in cell.polygons:
            if (poly.layer, poly.datatype) in [(1, 0), (1, 10)]:
                poly.layer, poly.datatype = 2, 0
    one = tmp_path / "one_layer.gds"
    lib.write_gds(str(one), max_points=8190)
    s = call(
        server(M.register_capacitance_ml_tools),
        "inspect_capacitance_layout",
        {"gds_path": str(one), "ground_layers": "2/0", "conductor_layers": "2/0"},
    )
    assert s.get("error") is None, s
    assert s["single_layer"] and s["ground_pieces"] and s["pieces"] and s["n_conductors"] == out["n_conductors"]
    assert "not found" in M.inspect_gds(str(tmp_path / "missing.gds"))["error"]


# ------------------------------------------------------------------------------------------------ dataset
class FakeDB:
    def __init__(self):
        from squadds.ml.capacitance.dataset import PlanarCapacitanceDB

        self.real = PlanarCapacitanceDB
        self.C = pd.DataFrame(MAXWELL, index=["G", "S", "N"], columns=["G", "S", "N"])

    def _bind(self, name, *a, **kw):
        bind(getattr(self.real, name), None, *a, **kw)

    def designs(self, **kw):
        self._bind("designs", **kw)
        ids = kw.get("design_ids") or []
        return pd.DataFrame({"design_id": [d for d in ids if d == "d1"], "family": ["xmon"] * ("d1" in ids)})

    def params(self, ids):
        self._bind("params", ids)
        return {"d1": {"cross_length": "180um"}}

    def simulations(self, ids):
        self._bind("simulations", ids)
        return pd.DataFrame({"design_id": ["d1"], "qc_ok": [True], "passes_completed": [12]})

    def label_passes(self, ids):
        self._bind("label_passes", ids)
        return pd.DataFrame({"design_id": ["d1", "d1"], "pass": [1, 2], "C_fF": [1.0, 1.1]})

    def labels(self, ids):
        self._bind("labels", ids)
        return {"d1": self.C}

    def optimized_physics_prior(self, ids):
        self._bind("optimized_physics_prior", ids)
        return {"d1": self.C}

    def physics_prior(self, ids):
        self._bind("physics_prior", ids)
        return {"d1": self.C}

    def gds_path(self, design_id):
        self._bind("gds_path", design_id)
        return Path("/tmp/d1.gds")

    def nearest(self, query, k=5, family=None, split=None):
        self._bind("nearest", query, k=k, family=family, split=split)
        return pd.DataFrame({"design_id": ["d2", "d3"][:k], "distance": [0.1, 0.2][:k]})


def test_design_record_and_nearest(monkeypatch):
    monkeypatch.setattr(M, "_dataset", lambda revision=None, local_dir=None: FakeDB())
    out = M.design_record("d1", parts=list(M.DESIGN_PARTS))
    assert "errors" not in out, out
    assert out["params"]["cross_length"] == "180um" and out["simulations"][0]["qc_ok"]
    assert out["label"]["names"] == ["G", "S", "N"] and out["physics_prior"]["matrix_fF"][0][1] == -20.0
    assert len(out["label_passes"]) == 2 and out["gds_path"].endswith("d1.gds")
    mcp = server(M.register_capacitance_ml_tools)
    d = call(mcp, "get_planar_capacitance_design", {"design_id": "d1"})
    assert set(M.DEFAULT_DESIGN_PARTS) <= set(d) and "gds_path" not in d
    assert "unknown design_id" in M.design_record("nope")["error"]
    assert "unknown parts" in M.design_record("d1", parts=["everything"])["error"]
    n = call(mcp, "nearest_planar_capacitance_designs", {"design_id": "d1", "k": 2, "split": "train"})
    assert [r["design_id"] for r in n["rows"]] == ["d2", "d3"] and "not a capacitance distance" in n["note"]


# ------------------------------------------------------------------------------------------------ Analyzer route
def test_closest_designs_ml(monkeypatch):
    import squadds.core.analysis as CA
    from squadds.ml.capacitance.bridge import MLSimulator

    real = CA.Analyzer
    closest = pd.DataFrame(
        {
            "design_id": ["a", "b"],
            "design_options": [{}, {}],
            "cross_to_claw": [5.0, 6.0],
            "cross_to_ground": [80.0, 90.0],
        },
        index=[3, 7],
    )
    seen = {}

    class FakeAnalyzer:
        def __init__(self, db):
            pass

        def find_closest(self, **kw):
            bind(real.find_closest, None, **kw)
            return closest

        def predict_capacitance(self, results_df=None, **kw):
            bind(real.predict_capacitance, None, results_df, **kw)
            bind(MLSimulator, **{k: v for k, v in kw.items() if k != "model"})
            seen.update(kw)
            r = {"warnings": [], "prediction": SimpleNamespace(threads=2, machine="m")}
            return pd.DataFrame(
                {
                    "cross_to_claw_ml": [5.1, 6.0],
                    "cross_to_ground_ml": [80.0, 92.0],
                    "maxwell_matrix_ml": [{}, {}],
                    "fidelity": ["ml:QCap"] * 2,
                    "ml_total_s": [1.0, 1.1],
                    "ml_result": [r, r],
                },
                index=closest.index,
            )

    class DB:
        def __getattr__(self, name):
            return lambda *a, **k: seen.setdefault(name, a)

    monkeypatch.setattr(CA, "Analyzer", FakeAnalyzer)
    monkeypatch.setattr(M, "_ml_missing", lambda: None)
    monkeypatch.setattr(M, "_predictor", lambda model, threads: object())
    out = AN.closest_designs_ml(DB(), "qubit", {"qubit_frequency_GHz": 4.5}, num_results=2, eps_r=9.8)
    assert out.get("error") is None, out
    assert seen["eps_r"] == 9.8 and seen["select_qubit"] == ("TransmonCross",)
    row = out["designs"][1]["stored_vs_ml"]["cross_to_ground"]
    assert row["diff_fF"] == pytest.approx(2.0) and row["diff_pct"] == pytest.approx(100 * 2 / 90)
    ag = out["agreement"]
    assert ag["entries"] == 4 and ag["mae_fF"] == pytest.approx(2.1 / 4)
    assert ag["rmse_fF"] == pytest.approx(np.sqrt((0.1**2 + 2.0**2) / 4))
    assert "error" in AN.closest_designs_ml(DB(), "qubit_cavity", {})


# ------------------------------------------------------------------------------------------------ two transmons
def test_coupled_transmon_tool_has_no_dead_z0():
    tools = {t.name: t for t in asyncio.run(server(D.register_capacitance_design_tools).list_tools())}
    assert "Z0" not in tools["solve_coupled_transmon_hamiltonian"].inputSchema["properties"]
    assert "Z0" in tools["capacitance_to_hamiltonian"].inputSchema["properties"]


@pytest.mark.parametrize(
    "row_f, row_kappa, c_end, f_line, claw_node, length, kappa",
    [  # the two-transmon example's resonator step, gap variant (A, B): rows, predicted row claws, our line frequency and claw node -> results
        (6.559985, 1603.519531, 61.7537, 6.9448, 77.1821, 8557.6252, 1616.1167),
        (6.517932, 1101.173828, 73.1770, 6.8129, 72.3607, 8729.7705, 1092.8325),
    ],
)
def test_resonator_correction_matches_reference(row_f, row_kappa, c_end, f_line, claw_node, length, kappa):
    from squadds.calcs.coupled_transmons import line_capacitance

    f_r = 6.6 if row_kappa > 1500 else 6.5
    node = claw_node + 1e15 * float(line_capacitance(f_line))
    out = call(
        server(D.register_capacitance_design_tools),
        "correct_half_wave_resonator",
        {
            "row_frequency_GHz": row_f,
            "row_kappa_kHz": row_kappa,
            "row_length_um": 8700.0,
            "f_r_GHz": f_r,
            "line_frequency_GHz": f_line,
            "node_capacitance_fF": node,
            "row_claw_capacitance_fF": c_end,
        },
    )
    assert out["resonator_length_um"] == pytest.approx(length, rel=2e-4)
    assert out["kappa_kHz"] == pytest.approx(kappa, rel=2e-4)
    assert "error" in D.resonator_correction(row_f, row_kappa, 8700.0, f_r, f_line, node)


# ------------------------------------------------------------------------------------------------ static checks
def _unused_parameters(module) -> list[str]:
    bad = []
    for fn in ast.walk(ast.parse(inspect.getsource(module))):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = fn.args
        params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs] + [x.arg for x in (a.vararg, a.kwarg) if x]
        body = fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        used = {n.id for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
        bad += [
            f"{module.__name__}.{fn.name}({p})" for p in params if p not in used and p not in ("self", "cls", "ctx")
        ]
    return bad


def test_every_tool_parameter_is_used():
    bad = [x for mod in (M, D, T, K, AN) for x in _unused_parameters(mod)]
    assert not bad, bad


TEXT_FILES = [
    *sorted((ROOT / "squadds_mcp" / "tools").glob("capacitance_*.py")),
    ROOT / "squadds_mcp" / "tools" / "analysis.py",
    ROOT / "squadds_mcp" / "resources" / "planar_capacitance_guide.py",
    ROOT / "squadds_mcp" / "resources" / "metadata.py",
    ROOT / "squadds_mcp" / "prompts" / "capacitance_workflows.py",
    ROOT / "MCP_README.md",
    ROOT / "docs" / "source" / "mcp_server.rst",
]
PCT = re.compile(r"(?<![\w.{])(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?) ?%")
WITH_ERRORS = re.compile(r"\s*\((?:MAE )?[\d.\-]+(?: fF)? ?/ ?(?:RMSE )?[\d.\-]+ fF\)")
NOT_ACCURACY = (  # percentages that are not capacitance accuracy claims
    "sensitivity (two-transmon example)",  # perturbation of the Hamiltonian, in MHz and %
    "move chi by ~",  # Hamiltonian effect of the cross couplings
    "of its target (J",  # Hamiltonian with the Q3D matrix, quoted in MHz
)


def test_every_accuracy_percentage_carries_mae_and_rmse():
    bad = []
    for f in TEXT_FILES:
        s = re.sub(r"\s+", " ", re.sub(r'"\s*\n\s*f?"', "", f.read_text(encoding="utf-8")))
        for m in PCT.finditer(s):
            after = s[m.end() : m.end() + 100]
            if WITH_ERRORS.match(after) or ("MAE" in after[:60] and "RMSE" in after):
                continue
            if any(p in s[max(0, m.start() - 160) : m.end() + 30] for p in NOT_ACCURACY):
                continue
            bad.append(f"{f.name}: ...{s[max(0, m.start() - 60) : m.end() + 40]}...")
    assert not bad, "\n".join(bad)
