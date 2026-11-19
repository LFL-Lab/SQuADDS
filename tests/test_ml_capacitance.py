"""Unit tests of squadds.ml.capacitance (CI-safe: synthetic GDS, untrained networks, mocked model hub)."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

import squadds.ml as ML

HAVE_ML = not ML.missing_ml_dependencies()
needs_ml = pytest.mark.skipif(not HAVE_ML, reason="squadds[ml] extra not installed")


def test_require_ml_names_missing_packages(monkeypatch):
    real = importlib.util.find_spec
    if "capacitance" in vars(ML):
        monkeypatch.delattr(ML, "capacitance")
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "torch" else real(n, *a))
    with pytest.raises(ImportError, match=r"pip install 'squadds\[ml\]'"):
        ML.require_ml()
    with pytest.raises(ImportError, match="torch"):
        ML.capacitance  # noqa: B018


def _tiny_gds(path: Path, gap_um: float = 20.0) -> Path:
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TINY")
    ground = gdstk.boolean(
        gdstk.rectangle((0, 0), (600, 600)), gdstk.rectangle((150, 150), (450, 450)), "not", layer=1, datatype=0
    )
    for p in ground:
        cell.add(p)
    cell.add(gdstk.rectangle((200, 280), (300 - gap_um / 2, 320), layer=1, datatype=10))
    cell.add(gdstk.rectangle((300 + gap_um / 2, 280), (400, 320), layer=1, datatype=10))
    lib.write_gds(str(path))
    return path


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("gdstk")
    return _tiny_gds(tmp_path_factory.mktemp("gds") / "tiny.gds")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fake_checkpoint(path: Path, seed: int) -> str:
    """An untrained full_gnn checkpoint in the released format (zero-initialised head: the network returns C^R)."""
    import torch

    from squadds.ml.capacitance._core import lp_surrogate as S
    from squadds.ml.capacitance._core import paper as PA
    from squadds.ml.capacitance.network import build_model

    torch.manual_seed(seed)
    nk, ek, gk = S.SCHEMA["nodekeys"], S.SCHEMA["edgekeys"], S.SCHEMA["globkeys"]
    m = build_model(len(nk), len(ek) + 1, len(gk), 2)
    scale = lambda n: {"median": [0.0] * n, "iqr": [1.0] * n}  # noqa: E731
    torch.save(
        {
            "state": m.state_dict(),
            "arm": "full_gnn",
            "rounds": 2,
            "scales": {"X": scale(len(nk)), "E": scale(len(ek) + 1), "G": scale(len(gk))},
            "schema": {
                "nodekeys": nk,
                "edgekeys": ek,
                "globkeys": gk,
                "physics": {"prior_method": "local_gap", "alpha": PA.ALPHA},
            },
            "closure_intercepts": {"0": 0.0, "1": -2.0},
            "direct_intercepts": {"0": 0.0, "1": 0.0},
        },
        path,
    )
    return _sha(path)


@pytest.fixture(scope="module")
def fake_bundle(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("bundle") / "QCap"
    (root / "qcap").mkdir(parents=True)
    shas = [_fake_checkpoint(root / "qcap" / f"s{s}.pt", s) for s in range(3)]
    return root, shas


@pytest.fixture
def fake_manifest(monkeypatch, fake_bundle):
    from squadds.ml.capacitance import models as M

    root, shas = fake_bundle
    man = copy.deepcopy(M.MANIFEST)
    e = man["models"]["qcap"]
    e["checkpoints"] = [{"name": f"s{s}", "path": f"qcap/s{s}.pt", "sha256": h} for s, h in enumerate(shas)]
    monkeypatch.setattr(M, "MANIFEST", man)
    monkeypatch.delenv(M.ENV_MODELS, raising=False)
    return root


@needs_ml
def test_manifest_lists_served_models():
    from squadds.ml.capacitance import list_models
    from squadds.ml.capacitance.models import MANIFEST, model_files

    rows = {r["model"]: r for r in list_models()}
    assert set(rows) == {"QCap", "QCap-Fast"}
    assert {r["key"] for r in rows.values()} == {"qcap", "qcap-fast"} and MANIFEST["default_model"] == "qcap"
    assert [m for m, r in rows.items() if r["default"]] == ["QCap"]
    assert "Use QCap for anything that matters" in rows["QCap-Fast"]["warning"]
    # the paper's held-out numbers are labelled as the paper's version, own fit is in-sample
    for m in ("QCap", "QCap-Fast"):
        r = rows[m]
        assert "belong to the paper's version" in r["reference"]["recipe"] and r["own"]["insample"]["mae_ff"] > 0
    assert (
        "improved release of the paper's fast model"
        in rows["QCap-Fast"]["title"] + MANIFEST["models"]["qcap-fast"]["summary"]
    )
    for k, e in MANIFEST["models"].items():
        assert e["repo_id"] == "SQuADDS/QCap"
        for rel, h in model_files(e):
            assert len(h) == 64 and rel.startswith(f"{k}/")
    assert len(model_files(MANIFEST["models"]["qcap-fast"])) == 4  # 3 networks + learned prior surrogate
    assert MANIFEST["prior_stacks"]["corrected"]["version"] == "v2m_sf_fc_dipfill_margin_slabfar"


@needs_ml
def test_load_model_from_path_and_env(fake_manifest, monkeypatch):
    from squadds.ml.capacitance import load_model
    from squadds.ml.capacitance.models import ENV_MODELS

    m = load_model("QCap", path=fake_manifest)
    assert m.name == "QCap" and m.key == "qcap" and m.root == fake_manifest and len(m.sha256) == 3
    for alias in ("qcap", "QCAP", " QCap "):
        assert load_model(alias, path=fake_manifest).name == "QCap"
    monkeypatch.setenv(ENV_MODELS, str(fake_manifest.parent))
    assert load_model().root == fake_manifest and load_model().name == "QCap"
    for gone in ("balanced", "paper", "qcap-lite"):  # not served: same error as any unknown name
        with pytest.raises(KeyError, match=r"unknown model .*available: \['QCap', 'QCap-Fast'\]"):
            load_model(gone)
    for old, new in (("flagship", "QCap"), ("fast", "QCap-Fast"), ("Flagship", "QCap")):
        with pytest.raises(ValueError, match=rf"renamed: use load_model\('{new}'\)"):
            load_model(old)
    with pytest.raises(TypeError, match=r"load_model\('QCap-Fast'\)"):
        load_model("QCap", tier="fast")


@needs_ml
def test_load_model_refuses_sha_mismatch(fake_manifest, tmp_path):
    from squadds.ml.capacitance import ModelIntegrityError, load_model

    bad = tmp_path / "bad"
    shutil.copytree(fake_manifest, bad)
    with open(bad / "qcap" / "s1.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ModelIntegrityError, match="sha256"):
        load_model("QCap", path=bad)
    (bad / "qcap" / "s2.pt").unlink()
    with pytest.raises(ModelIntegrityError):
        load_model("QCap", path=bad)


@needs_ml
def test_load_model_downloads_from_hub_at_pinned_revision(fake_manifest, tmp_path, monkeypatch):
    import huggingface_hub

    from squadds.ml.capacitance import load_model

    calls = []

    def fake_download(repo_id, filename, revision=None, cache_dir=None, local_files_only=False):
        calls.append((repo_id, filename, revision))
        if filename == "config.json":
            raise FileNotFoundError(filename)
        dst = tmp_path / "snapshots" / str(revision) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fake_manifest / filename, dst)
        return str(dst)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    m = load_model("QCap", revision="abc123")
    assert m.root == tmp_path / "snapshots" / "abc123" and m.revision == "abc123"
    assert {c[0] for c in calls} == {"SQuADDS/QCap"} and {c[2] for c in calls} == {"abc123"}
    assert calls[0][1] == "config.json"
    assert sorted(c[1] for c in calls[1:]) == [f"qcap/s{s}.pt" for s in range(3)]


@needs_ml
def test_prepare_inputs_and_hash(tiny):
    import gdstk

    from squadds.ml.capacitance import prepare_layout
    from squadds.ml.capacitance._runtime import layout_hash

    a = prepare_layout(tiny)
    b = prepare_layout(gdstk.read_gds(str(tiny)))
    assert a.names == ["C1", "C2"] and b.names == a.names
    assert layout_hash(a) == layout_hash(b) == layout_hash(prepare_layout(a))
    with pytest.raises(TypeError):
        prepare_layout(3.0)


@needs_ml
def test_prepared_layout_plot(tiny):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from squadds.ml.capacitance import prepare_layout

    lay = prepare_layout(tiny)
    ax = lay.plot()
    assert [t.get_text() for t in ax.texts] == ["C1", "C2"] and ax.get_aspect() == 1.0
    _, ax2 = plt.subplots()
    assert lay.plot(ax=ax2, title="mine") is ax2 and ax2.get_title() == "mine"
    assert [t.get_text() for t in lay.plot(labels={"C1": "pad"}).texts] == ["pad", "C2"]
    assert not lay.plot(labels=False).texts
    plt.close("all")


@needs_ml
def test_embedding_shapes_and_names(tiny):
    from squadds.layouts.capacitance_graph import EMBEDDING_VERSION, encode

    emb = encode(tiny)
    assert emb["version"] == EMBEDDING_VERSION == "planar-capacitance-graph-v1"
    assert emb["names"] == ["G", "C1", "C2"]
    assert emb["node_features"].shape == (3, 60) and len(emb["node_feature_names"]) == 60
    assert emb["pair_features"].shape == (3, 42) and len(emb["pair_feature_names"]) == 42
    assert emb["global_features"].shape == (4,) and emb["global_feature_names"][0] == "log_chip_area"
    assert emb["pairs"].tolist() == [[0, 1], [0, 2], [1, 2]]
    assert len(emb["code_sha256"]) == 64
    np.testing.assert_array_equal(encode(tiny)["pair_features"], emb["pair_features"])


@needs_ml
def test_physics_priors(tiny):
    from squadds.ml.capacitance import optimized_physics_prior, physics_prior
    from squadds.ml.capacitance.result import physical_checks

    c = physics_prior(tiny)
    o = optimized_physics_prior(tiny)
    for r in (c, o):
        assert r.names == ["G", "C1", "C2"] and r.C_fF.shape == (3, 3)
        assert physical_checks(r.C_fF)["offdiag_all_negative"] and np.allclose(r.C_fF, r.C_fF.T)
        assert r.prior_version == "v2m_sf_fc_dipfill_margin_slabfar" and r.threads == _rt().default_threads()
    assert c.levels_ppg == [0.375] and sorted(o.solves_fF) == [1.5, 3.0]
    lo, hi = -o.solves_fF[1.5][1, 2], -o.solves_fF[3.0][1, 2]
    assert -o.C_fF[1, 2] == pytest.approx(np.exp(2 * np.log(hi) - np.log(lo)), rel=1e-12)
    assert 0.01 < -o.C_fF[1, 2] < 100  # fF, two 100 x 40 µm pads 20 µm apart on silicon


@needs_ml
def test_predictor_end_to_end_untrained_network(fake_manifest, tiny):
    """Zero-initialised head: the predicted mutuals are the optimized prior's (float32 base), the closure gives the self terms."""
    from squadds.ml.capacitance import CapacitancePredictor, load_model, optimized_physics_prior

    p = CapacitancePredictor(load_model("QCap", path=fake_manifest))
    r = p.predict(tiny)
    o = optimized_physics_prior(tiny)
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(r.matrix_fF[off], o.C_fF[off], rtol=1e-6)
    assert r.names == ["G", "C1", "C2"] and r.checks["passed"] and r.model == "QCap"
    assert (
        r.prior_version == "v2m_sf_fc_dipfill_margin_slabfar"
        and r.weights_sha256 == load_model("QCap", path=fake_manifest).weights_sha256
    )
    assert {"prepare_s", "geometry_s", "prior_s", "network_s", "total_s"} <= set(
        r.timing_s
    ) and r.threads == _rt().default_threads()
    again = p.predict(tiny)
    assert again.cached and np.array_equal(again.matrix_fF, r.matrix_fF)
    batch = p.predict([tiny, tiny])
    assert len(batch) == 2 and all(np.array_equal(b.matrix_fF, r.matrix_fF) for b in batch)


@needs_ml
def test_predictor_calibrated_substrate(fake_manifest, tiny):
    """Off the training stack the prior is solved at the requested stack (untrained head: mutuals = that stack's prior)."""
    import warnings

    from squadds.ml.capacitance import CapacitancePredictor, load_model, optimized_physics_prior

    p = CapacitancePredictor(load_model("QCap", path=fake_manifest))
    base = p.predict(tiny)
    off = ~np.eye(3, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = p.predict(tiny, eps_r=11.45, thickness_um=50)  # tested: Si at 10-1000 µm
    o = optimized_physics_prior(tiny, eps_r=11.45, thickness_um=50)
    np.testing.assert_allclose(r.matrix_fF[off], o.C_fF[off], rtol=1e-6)
    assert not np.allclose(r.matrix_fF[off], base.matrix_fF[off], rtol=1e-3)
    sub = r.substrate
    assert sub["route"].startswith("calibrated") and sub["tested"] and sub["eps_rescale_factor"] is None
    assert sub["prior_solved_at"] == {"eps_r": 11.45, "thickness_um": 50.0}
    assert sub["embedding_stack"] == {"eps_r": 11.45, "thickness_um": 350.0} and "bounded-error" in sub["caveat"]
    assert o.stack["substrate"] == {"eps_r": 11.45, "thickness_um": 50.0, "tested": True}
    with pytest.warns(UserWarning, match="outside the tested range"):
        u = p.predict(tiny, eps_r=9.8, thickness_um=500)
    assert not u.substrate["tested"] and any("untested" in w for w in u.warnings)
    with pytest.warns(UserWarning, match="outside the tested range"):
        o2 = optimized_physics_prior(tiny, eps_r=9.8, thickness_um=500)
    np.testing.assert_allclose(u.matrix_fF[off], o2.C_fF[off], rtol=1e-6)
    again = p.predict(tiny)  # the prior cache is keyed by the stack
    assert np.array_equal(again.matrix_fF, base.matrix_fF) and again.substrate["route"] == "training stack"
    with pytest.raises(ValueError, match="thickness"):
        p.predict(tiny, thickness_um=0)
    with pytest.raises(TypeError):
        p.predict(tiny, eps_r=9.8, substrate_rescale="quasi_static")


@needs_ml
def test_training_stack_prior_path_is_unchanged(tiny):
    """The prior solved with the training stack passed explicitly is bit-identical to the default path (scene stack)."""
    from squadds.ml.capacitance import _runtime as RT
    from squadds.ml.capacitance._core import api as A
    from squadds.ml.capacitance._core import paper as PA
    from squadds.ml.capacitance.priors import _registry

    lay = RT.prepare_layout(tiny)
    with RT.LOCK:
        RT.configure(2)
        PA.apply_stack(_registry(), {"stack": "corrected"}, 2)
        sc = A.scene_from_prepared(lay)
        a, _ = PA.solve_levels(sc, [1.5, 3.0], 2)
        b, _ = PA.solve_levels(sc, [1.5, 3.0], 2, (RT.TRAIN_EPS_R, RT.TRAIN_THICKNESS_UM))
    assert all(np.array_equal(a[k], b[k]) for k in (1.5, 3.0))
    assert sc.stack["substrate_eps_r"] == 11.45 and sc.stack["substrate_thickness_um"] == 350.0


@needs_ml
def test_qcap_fast_substrate_fallback():
    """QCap-Fast (learned prior surrogate) cannot solve a prior: off the stack it is eps-rescaled, recorded and warned."""
    from squadds.ml.capacitance.predictor import _substrate

    stack, sub, w = _substrate(9.8, 350.0, True)
    assert sub["route"] == "fast: eps-rescaled training-stack prediction; thickness not modelled"
    assert (
        sub["eps_rescale_factor"] == pytest.approx((1 + 9.8) / (1 + 11.45), rel=1e-15)
        and sub["prior_solved_at"] is None
    )
    assert (
        w
        and w[0].startswith("QCap-Fast does not solve")
        and "use QCap for other substrates" in w[0]
        and stack == (9.8, 350.0)
    )
    stack, sub, w = _substrate(11.45, 350.0, True)
    assert stack is None and sub["eps_rescale_factor"] == 1.0 and not w


def test_result_checks_dataframe_and_sim_results():
    from squadds.ml.capacitance.result import CapacitanceResult

    C = np.array([[30.0, -5.0, -6.0], [-5.0, 8.0, -2.0], [-6.0, -2.0, 9.0]])
    r = CapacitanceResult(names=["G", "C1", "C2"], matrix_fF=C, model="QCap")
    assert r.checks["passed"] and r.checks["symmetric"] and r.checks["closure"]
    sim = r.to_sim_results(names={"C1": "north", "C2": "south"})
    assert sim["north_to_north"] == 8.0 and sim["north_to_south"] == 2.0 and sim["north_to_ground"] == 5.0
    assert sim["south_to_ground"] == 6.0 and sim["ground_to_ground"] == 30.0 and sim["units"] == "fF"
    assert (
        sim["maxwell_matrix"]["C_north_south"] == -2.0
        and sim["maxwell_matrix"]["C_G_G"] == 30.0
        and len(sim["maxwell_matrix"]) == 9
    )
    if importlib.util.find_spec("pandas"):
        df = r.to_dataframe()
        assert list(df.columns) == ["G", "C1", "C2"] and df.loc["C1", "C2"] == -2.0
    bad = CapacitanceResult(names=["G", "C1", "C2"], matrix_fF=C * np.array([1, 1, -1]), model="m")
    assert not bad.checks["passed"]
    from squadds.ml.capacitance.result import physical_warning

    assert physical_warning(r.checks) is None
    w = physical_warning(bad.checks)
    assert w.startswith("physical check failed") and "offdiag all negative" in w and "do not use it" in w


def _one_layer(src, dst, layer=(2, 0)):
    """Every polygon of src moved to one layer, file order and vertices kept."""
    import gdstk

    lib = gdstk.read_gds(str(src))
    for c in lib.cells:
        for q in c.polygons:
            q.layer, q.datatype = layer
    lib.write_gds(str(dst), max_points=8190)
    return dst


@needs_ml
def test_one_layer_ground_detection(tiny, tmp_path):
    """Ground and conductors on one layer: the ground is detected (the piece that encloses the rest), the split is the two-layer
    one, ground_pieces overrides it and bad numbers fail clearly."""
    from squadds.ml.capacitance import inspect_layout, prepare_layout
    from squadds.ml.capacitance._core.layout import LayoutError
    from squadds.ml.capacitance._runtime import layout_hash

    one = _one_layer(tiny, tmp_path / "one.gds")
    info = inspect_layout(one)  # single-layer file, default layer arguments
    assert (
        info["detected"]
        and info["ground_pieces"] == [3]
        and info["confidence"] == "high"
        and info["names"] == ["C1", "C2"]
    )
    assert [(p["role"], p["name"]) for p in info["pieces"]] == [
        ("conductor", "C1"),
        ("conductor", "C2"),
        ("ground", "G"),
    ]
    assert info["figure"].axes and any("ground_pieces=[piece numbers]" in w for w in info["warnings"])
    same = {"ground_layers": [(2, 0)], "conductor_layers": [(2, 0)]}
    assert layout_hash(prepare_layout(one, **same)) == layout_hash(prepare_layout(tiny))
    assert prepare_layout(one, ground_pieces=[3], **same).info["ground_detection"]["confidence"] == "user"
    for bad, msg in (([4], "pieces 1..3"), ([1, 2, 3], "every piece")):
        with pytest.raises(LayoutError, match=msg):
            prepare_layout(one, ground_pieces=bad, **same)
    with pytest.raises(LayoutError, match="share one layer"):
        prepare_layout(tiny, ground_pieces=[1])
    assert not inspect_layout(tiny)["detected"]


def test_caches_bounded_by_count_and_bytes(monkeypatch):
    from collections import OrderedDict

    from squadds.ml.capacitance import predictor as PD

    c = OrderedDict()
    monkeypatch.setattr(PD, "_CACHE_SIZE", [3])
    monkeypatch.setattr(PD, "_CACHE_BYTES", [4000])
    monkeypatch.setattr(PD, "_NBYTES", {})
    for k in range(5):
        PD._put(c, k, {"P": np.zeros(100)})  # ~1 kB each (800 bytes of array)
    assert list(c) == [2, 3, 4]  # count bound
    PD._get(c, 2)
    PD._put(c, 5, np.zeros(300))  # 2400 bytes: the least recently used entries go first
    assert list(c) == [2, 5] and sum(PD._NBYTES[id(c)].values()) <= 4000
    PD._put(c, 6, np.zeros(1000))  # larger than the bound: not kept
    assert not c and not PD._NBYTES[id(c)]


@needs_ml
def test_one_layer_prediction_bitwise(fake_manifest, tiny, tmp_path, capsys):
    from squadds.ml.capacitance import CapacitancePredictor, load_model
    from squadds.ml.capacitance.cli import main
    from squadds.ml.capacitance.predictor import clear_cache

    one = _one_layer(tiny, tmp_path / "one.gds")
    p = CapacitancePredictor(load_model("QCap", path=fake_manifest))
    clear_cache()
    a = p.predict(tiny)
    clear_cache()
    with pytest.warns(UserWarning, match="detected automatically as piece #3"):
        b = p.predict(one)
    assert a.matrix_fF.tobytes() == b.matrix_fF.tobytes() and b.names == a.names and not b.cached
    assert b.ground_detection["ground_pieces"] == [3] and a.ground_detection is None
    assert (
        main(
            [
                "predict",
                str(one),
                "--model-path",
                str(fake_manifest),
                "--ground",
                "2/0",
                "--conductor",
                "2/0",
                "--ground-pieces",
                "3",
            ]
        )
        == 0
    )
    assert ",G,C1,C2" in capsys.readouterr().out
    with pytest.raises(Exception, match="pieces 1..3"):
        main(["predict", str(one), "--model-path", str(fake_manifest), "--ground-pieces", "7"])


@needs_ml
def test_cli_predict_csv(fake_manifest, tiny, capsys):
    from squadds.ml.capacitance.cli import main

    assert main(["predict", str(tiny), "--model-path", str(fake_manifest)]) == 0
    out = capsys.readouterr().out
    assert "# " in out and ": QCap," in out and ",G,C1,C2" in out
    with pytest.raises(SystemExit):
        main(["predict", str(tiny), "--model-path", str(fake_manifest), "--substrate-rescale", "quasi_static"])


def _stdout_on_tty(code: str) -> bytes:
    import os
    import select
    import subprocess
    import sys

    m, s = os.openpty()
    p = subprocess.Popen([sys.executable, "-c", code], stdout=s, stderr=subprocess.DEVNULL)
    os.close(s)
    out = b""
    while True:
        if select.select([m], [], [], 0.5)[0]:
            try:
                b = os.read(m, 1 << 16)
            except OSError:
                break
            if not b:
                break
            out += b
        elif p.poll() is not None:
            break
    p.wait(timeout=600)
    os.close(m)
    return out


@pytest.mark.skipif(not hasattr(__import__("os"), "openpty"), reason="needs a pseudo-terminal")
def test_cli_output_has_no_exit_ansi_reset():
    pytest.importorskip("colorama")
    from squadds.ml.capacitance import cli

    armed = "import colorama; colorama.init(); "
    assert _stdout_on_tty(armed + "print('a,b')").endswith(b"\x1b[0m")
    load = (
        f"import importlib.util as u; s = u.spec_from_file_location('c', {cli.__file__!r}); c = u.module_from_spec(s); "
    )
    fixed = _stdout_on_tty(armed + load + "s.loader.exec_module(c); c._no_exit_ansi_reset(); print('a,b')")
    assert fixed.endswith(b"a,b\r\n") and b"\x1b" not in fixed
    assert "_no_exit_ansi_reset()\n    if a.format" in Path(cli.__file__).read_text(encoding="utf-8")


def test_core_port_matches_sources_json():
    """Every ported file still has the sha256 recorded at port time (numerics of the frozen stage untouched)."""
    import json

    root = Path(ML.__file__).parent / "capacitance" / "_core"
    files = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))["files"]
    assert len(files) >= 30
    for rel, e in files.items():
        if rel.startswith("../"):
            continue
        assert _sha(root / rel) == e["sha256"], rel
        assert all(
            r.get("kind") == "user_text"  # user-facing strings only, e.g. a model that is not served
            or "import" in r["replacement"]
            or "boot" in r["replacement"]
            or "here =" in r["replacement"]
            or "squadds" in r["replacement"]
            for r in e["rewrites"]
        ), rel


def test_public_exports_resolve():
    import squadds.ml.capacitance as C

    for name in C.__all__:
        assert getattr(C, name) is not None
    assert {"MLSimulator", "PlanarCapacitanceDB", "fit_in_family", "contribute_capacitance"} <= set(C.__all__)


def test_default_threads(monkeypatch):
    """Default threads: usable CPUs, at most one per 2.5 GB of RAM and 32; SQUADDS_CAPML_THREADS and threads= override."""
    RT = _rt()
    monkeypatch.delenv("SQUADDS_CAPML_THREADS", raising=False)
    monkeypatch.setattr(RT, "_usable_cpus", lambda: 18)
    monkeypatch.setattr(RT, "_total_ram_gb", lambda: 48.0)
    assert _rt().default_threads() == 18
    monkeypatch.setattr(RT, "_total_ram_gb", lambda: 16.0)
    assert _rt().default_threads() == 6
    monkeypatch.setattr(RT, "_total_ram_gb", lambda: None)
    monkeypatch.setattr(RT, "_usable_cpus", lambda: 128)
    assert _rt().default_threads() == RT.MAX_DEFAULT_THREADS == 32
    monkeypatch.setattr(RT, "_total_ram_gb", lambda: 1.0)
    assert _rt().default_threads() == 1
    monkeypatch.setenv("SQUADDS_CAPML_THREADS", "3")
    assert _rt().default_threads() == 3 and RT.resolve_threads(None) == 3 and RT.resolve_threads(5) == 5


def _rt():
    from squadds.ml.capacitance import _runtime

    return _runtime
