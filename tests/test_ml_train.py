"""Unit tests of squadds.ml.capacitance training, shard data and scoring (CI-safe: tiny synthetic shards, a few steps)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import squadds.ml as ML

pytestmark = pytest.mark.skipif(bool(ML.missing_ml_dependencies()), reason="squadds[ml] extra not installed")

FAST = dict(steps=30, log_every=10)


def _layout(rng, n, key, family, split, schema):
    names = ["G"] + [f"C{i}" for i in range(1, n)]
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    m = np.exp(rng.normal(0, 1, len(pairs)))
    C = np.zeros((n, n))
    C[pairs[:, 0], pairs[:, 1]] = -m
    C[pairs[:, 1], pairs[:, 0]] = -m
    C[np.diag_indices(n)] = -C.sum(1) + rng.uniform(0.01, 0.1, n)
    nk, ek = schema["nodekeys"], schema["edgekeys"]
    X = rng.normal(0, 1, (n, len(nk))).astype(np.float32)
    X[:, nk.index("role_ground")] = [1.0] + [0.0] * (n - 1)
    X[:, -1] = 0.0
    E = rng.normal(0, 1, (len(pairs), len(ek))).astype(np.float32)
    E[:, ek.index("is_ground_pair")] = (pairs[:, 0] == 0).astype(np.float32)
    pri = np.stack([m * np.exp(rng.normal(0, 0.02, len(m))), m * np.exp(rng.normal(0, 0.01, len(m)))], -1)
    return dict(
        key=key,
        split=split,
        family=family,
        names=names,
        X=X,
        E=E,
        G=rng.normal(0, 1, 4).astype(np.float32),
        C=C.astype(np.float32),
        sig=np.array([x != "G" for x in names]),
        pri=pri.astype(np.float32),
    )


def _tiny_gds(path: Path, gap_um: float = 20.0) -> Path:
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TINY")
    for p in gdstk.boolean(
        gdstk.rectangle((0, 0), (600, 600)), gdstk.rectangle((150, 150), (450, 450)), "not", layer=1
    ):
        cell.add(p)
    cell.add(gdstk.rectangle((200, 280), (300 - gap_um / 2, 320), layer=1, datatype=10))
    cell.add(gdstk.rectangle((300 + gap_um / 2, 280), (400, 320), layer=1, datatype=10))
    lib.write_gds(str(path))
    return path


def make_shards(path, n_layouts=24, seed=0, families=("fa", "fb"), prefix="k", val_every=4):
    from squadds.ml.capacitance.training_data import base_schema, write_shards

    rng = np.random.default_rng(seed)
    schema = base_schema()
    rows = [
        _layout(
            rng,
            2 + (b % 3),
            f"{prefix}{b:03d}",
            families[b % len(families)],
            "val" if b % val_every == 0 else "train",
            schema,
        )
        for b in range(n_layouts)
    ]
    return write_shards(rows, path, schema)


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    return make_shards(tmp_path_factory.mktemp("shards") / "data")


@pytest.fixture(scope="module")
def run(shards, tmp_path_factory):
    from squadds.ml.capacitance.train import TrainConfig, fit

    out = tmp_path_factory.mktemp("runs") / "s0"
    fit(shards, out, seed=0, config=TrainConfig(**FAST), threads=1, verbose=False)
    return out


# ---------------------------------------------------------------------------------------------------------------- labels
def _good():
    return np.array([[10.0, -3.0, -4.0], [-3.0, 5.0, -1.0], [-4.0, -1.0, 6.0]]), ["G", "N", "S"]


def test_label_qc_accepts_maxwell_matrix():
    from squadds.ml.capacitance.training_data import label_qc

    C, names = _good()
    assert label_qc(C, names)["passed"]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda C: C.__setitem__((0, 1), -3.5), "asymmetric"),
        (lambda C: (C.__setitem__((1, 2), 0.5), C.__setitem__((2, 1), 0.5)), "positive off-diagonal"),
        (lambda C: C.__setitem__((1, 1), -1.0), "non-positive diagonal"),
        (lambda C: C.__setitem__((1, 1), 3.5), "closure"),
        (lambda C: C.__setitem__((2, 2), np.nan), "non-finite"),
    ],
)
def test_label_qc_refuses(mutate, match):
    from squadds.ml.capacitance.training_data import LabelQCError, label_qc

    C, names = _good()
    mutate(C)
    with pytest.raises(LabelQCError, match=match):
        label_qc(C, names)


def test_label_qc_refuses_non_pd_and_ground_order():
    from squadds.ml.capacitance.training_data import LabelQCError, label_qc

    C = np.array([[10.0, -3.0, -4.0], [-3.0, 4.0, -4.5], [-4.0, -4.5, 9.0]])
    with pytest.raises(LabelQCError, match="closure|positive definite"):
        label_qc(C, ["G", "N", "S"])
    with pytest.raises(LabelQCError, match="ground"):
        label_qc(_good()[0], ["N", "G", "S"])


def test_read_q3d_csv_last_complete_pass_and_units(tmp_path):
    from squadds.ml.capacitance.training_data import read_q3d_csv

    p = tmp_path / "cap_matrix.csv"
    hdr = ['"Freq [GHz]"', '"Pass []"'] + [f'"C({a},{b}) [pF]"' for a in ("N", "G") for b in ("N", "G")]
    p.write_text(
        ",".join(hdr) + "\n13,1,0.004,-0.003,-0.003,0.010\n13,2,0.005,-0.0031,-0.0031,0.011\n13,3,nan,nan,nan,nan\n",
        encoding="utf-8",
    )
    names, C, info = read_q3d_csv(p)
    assert names == ["G", "N"]
    assert np.allclose(C, [[11.0, -3.1], [-3.1, 5.0]])
    assert info["label_pass_index"] == 1 and info["complete_passes"] == 2
    q = tmp_path / "cap_matrix_full.csv"
    hdr = ['"Freq [GHz]"', '"Pass []"'] + [f'"C({a},{b})*1e15 [pF]"' for a in ("G", "N") for b in ("G", "N")]
    q.write_text(",".join(hdr) + "\n13,1,11e12,-3.1e12,-3.1e12,5e12\n", encoding="utf-8")
    assert np.allclose(read_q3d_csv(q)[1], [[11.0, -3.1], [-3.1, 5.0]])


# ---------------------------------------------------------------------------------------------------------------- shards
def test_shard_layout(shards):
    from squadds.ml.capacitance.training_data import ARRAYS, read_names, read_schema, read_shards, shard_summary

    ds = read_shards(shards)
    assert [d["n"] for d in ds] == [2, 3, 4]
    for d in ds:
        assert set(ARRAYS) <= set(d)
        assert d["X"].shape[-1] == 61 and d["E"].shape[-1] == 42 and d["pri"].shape[-1] == 2
        assert np.array_equal(d["pairs"], np.array([(i, j) for i in range(d["n"]) for j in range(i + 1, d["n"])]))
    s = read_schema(shards)
    assert s["nodekeys"][-1] == "missing_log_min_halfwidth" and s["physics"]["alpha"] > 0
    assert len(read_names(shards)) == 24 and shard_summary(shards)["layouts"] == 24


def test_merge_shards(shards, tmp_path):
    from squadds.ml.capacitance.training_data import merge_shards, read_shards

    user = make_shards(tmp_path / "user", n_layouts=6, seed=1, families=("mine",), prefix="u")
    out = merge_shards(shards, user, tmp_path / "merged", all_split="train")
    ds = read_shards(out)
    keys = [str(k) for d in ds for k in d["key"]]
    assert len(keys) == 30 and len(set(keys)) == 30
    assert all((d["split"] == "train").all() for d in ds)
    with pytest.raises(ValueError, match="duplicate"):
        merge_shards(shards, shards, tmp_path / "dup")
    s = json.loads((user / "schema.json").read_text(encoding="utf-8"))
    s["physics"]["prior_version"] = "other"
    (user / "schema.json").write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(ValueError, match="prior version"):
        merge_shards(shards, user, tmp_path / "mismatch")


def test_merge_fills_base_only_string_arrays(shards, tmp_path):
    from squadds.ml.capacitance.training_data import merge_shards, read_shards, subset_shards

    def add_origin(d):
        return np.ones(len(d["key"]), bool)

    base = subset_shards(shards, tmp_path / "base", keep=add_origin)
    for p in base.glob("n*.npz"):
        d = dict(np.load(p))
        d["origin"], d["group"] = d["split"].copy(), d["family"].copy()
        np.savez_compressed(p, **d)
    user = make_shards(tmp_path / "user", n_layouts=3, seed=2, families=("mine",), prefix="u")
    ds = read_shards(merge_shards(base, user, tmp_path / "m", user_split="train"))
    origin = np.concatenate([d["origin"] for d in ds])
    group = np.concatenate([d["group"] for d in ds])
    assert (origin == "user").sum() == 3 and (group == "mine").sum() == 3


def test_subset_shards(shards, tmp_path):
    from squadds.ml.capacitance.training_data import read_shards, subset_shards

    sub = subset_shards(shards, tmp_path / "sub", train_keys=["k001", "k002"], val_keys=["k003"])
    rows = {str(k): str(s) for d in read_shards(sub) for k, s in zip(d["key"], d["split"])}
    assert rows == {"k001": "train", "k002": "train", "k003": "val"}
    with pytest.raises(KeyError):
        subset_shards(shards, tmp_path / "sub2", train_keys=["nope"])


def test_shards_from_model_input(shards, tmp_path):
    import pandas as pd

    from squadds.ml.capacitance.training_data import read_names, read_shards, shards_from_model_input

    names = read_names(shards)
    mi, lab = [], []
    for d in read_shards(shards):
        for b, k in enumerate(d["key"]):
            k = str(k)
            mi.append(
                dict(
                    design_id=k,
                    n_conductors=d["n"],
                    net_order=names[k],
                    model_family=str(d["family"][b]),
                    X=d["X"][b].ravel(),
                    E=d["E"][b].ravel(),
                    G=d["G"][b],
                    pri=d["pri"][b].ravel(),
                    sig=d["sig"][b],
                    prior_version="v2m_sf_fc_dipfill_margin_slabfar_R1.5_3",
                    split=str(d["split"][b]),
                )
            )
            for i in range(d["n"]):
                for j in range(i, d["n"]):
                    lab.append(dict(design_id=k, a_idx=i, b_idx=j, C_fF=float(d["C"][b][i, j])))
    out = shards_from_model_input(pd.DataFrame(mi), pd.DataFrame(lab), tmp_path / "mi")
    a = {str(k): d["C"][b] for d in read_shards(shards) for b, k in enumerate(d["key"])}
    b = {str(k): d["C"][j] for d in read_shards(out) for j, k in enumerate(d["key"])}
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    with pytest.raises(PermissionError):
        shards_from_model_input(pd.DataFrame(mi), pd.DataFrame(lab), tmp_path / "t", splits=("train", "test"))


# ---------------------------------------------------------------------------------------------------------------- fit
def test_train_config_defaults():
    from squadds.ml.capacitance.train import TrainConfig

    c = TrainConfig()
    assert (c.steps, c.lr, c.weight_decay, c.lr_min, c.grad_clip) == (18000, 1e-3, 1e-4, 1e-5, 5.0)
    assert (c.huber_delta, c.w_signal_self, c.w_ground_self, c.max_batch, c.pair_budget) == (0.05, 0.1, 0.02, 64, 8192)


def test_fit_checkpoint(run, shards):
    import torch

    from squadds.ml.capacitance.train import TrainConfig, release_eligible

    ck = torch.load(run / "model.pt", weights_only=False)
    for k in (
        "state",
        "arm",
        "rounds",
        "scales",
        "schema",
        "closure_intercepts",
        "direct_intercepts",
        "train_keys",
        "data_sha",
        "code_sha",
        "machine",
        "device",
        "threads",
        "package_version",
        "bit_identical_to_release",
        "config",
    ):
        assert k in ck, k
    assert ck["arm"] == "full_gnn" and ck["rounds"] == 2 and ck["steps"] == 30
    assert len(ck["train_keys"]) == 18 and len(ck["scales"]["E"]["median"]) == 43
    assert ck["bit_identical_to_release"] is False and not release_eligible("cpu", 1, TrainConfig())
    assert ck["release_environment_match"] is False and ck["release_environment_diff"]
    assert ck["environment"]["threads"] == 1 and ck["release_environment"]["machine"] == "aarch64"
    r = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert r["val"]["layouts"] == 6 and np.isfinite(r["val"]["median_pct"])
    assert (run / "metrics.csv").exists()
    assert r["release_environment_match"] is False and r["bit_identical_to_release"] is False


def test_release_environment_rule():
    from squadds.ml.capacitance.train import (
        RELEASE_ENVIRONMENT,
        TrainConfig,
        release_eligible,
        release_environment_diff,
    )

    rel = {k: RELEASE_ENVIRONMENT[k] for k in ("machine", "system", "device", "threads", "torch", "numpy")}
    assert rel["machine"] == "aarch64" and rel["device"] == "cpu" and rel["threads"] == 2
    assert release_eligible("cpu", 2, TrainConfig(), env=rel)
    assert release_eligible("cpu", 2, TrainConfig(), env={**rel, "torch": rel["torch"].split("+")[0] + "+cpu"})
    x86 = {**rel, "machine": "x86_64"}
    assert not release_eligible("cpu", 2, TrainConfig(), env=x86)
    assert release_environment_diff("cpu", 2, TrainConfig(), env=x86) == ["machine: 'x86_64' (release 'aarch64')"]
    assert not release_eligible("cpu", 2, TrainConfig(), env={**rel, "numpy": "1.26.4"})
    assert not release_eligible("cpu", 2, TrainConfig(), env={**rel, "torch": "2.15.0"})
    assert not release_eligible("cpu", 2, TrainConfig(), env={**rel, "machine": "arm64", "system": "Darwin"})
    assert not release_eligible("cpu", 2, TrainConfig(), env={**rel, "threads": 1})
    assert not release_eligible("cpu", 2, TrainConfig(steps=10), env=rel)


def test_fit_is_deterministic(shards, run, tmp_path):
    import torch

    from squadds.ml.capacitance.train import TrainConfig, fit

    fit(shards, tmp_path / "again", seed=0, config=TrainConfig(**FAST), threads=1, verbose=False, validate=False)
    a = torch.load(run / "model.pt", weights_only=False)["state"]
    b = torch.load(tmp_path / "again" / "model.pt", weights_only=False)["state"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_fit_masks_nan_labels(tmp_path):
    import torch

    from squadds.ml.capacitance.train import TrainConfig, fit

    data = make_shards(tmp_path / "d", n_layouts=12, seed=3)
    for p in data.glob("n*.npz"):
        d = dict(np.load(p))
        d["C"][0, 0, 1] = d["C"][0, 1, 0] = np.nan
        d["C"][0, 0, 0] = np.nan
        np.savez_compressed(p, **d)
    fit(data, tmp_path / "r", config=TrainConfig(**FAST), threads=1, verbose=False, validate=False)
    st = torch.load(tmp_path / "r" / "model.pt", weights_only=False)["state"]
    assert all(torch.isfinite(v).all() for v in st.values())


@pytest.mark.parametrize("arm", ["direct_gnn", "pair_mlp"])
def test_fit_other_arms(shards, tmp_path, arm):
    from squadds.ml.capacitance.train import TrainConfig, fit

    r = fit(shards, tmp_path / arm, arm=arm, config=TrainConfig(steps=5, log_every=5), threads=1, verbose=False)
    assert np.isfinite(r["val"]["median_pct"])


def test_fit_seeds_and_trained_model(shards, tmp_path):
    from squadds.ml.capacitance.train import TrainConfig, fit_seeds, trained_model

    runs = fit_seeds(
        shards, tmp_path / "seeds", seeds=(0, 1), config=TrainConfig(steps=5, log_every=5), threads=1, validate=False
    )
    assert [r["seed"] for r in runs] == [0, 1]
    m = trained_model([tmp_path / "seeds" / "s0", tmp_path / "seeds" / "s1"], name="t")
    assert len(m.weights_sha256) == 2 and m.entry["stack"] == "corrected"
    assert all((Path(m.root) / c["path"]).exists() for c in m.entry["checkpoints"])


def test_trained_model_predicts_gds(run, tmp_path):
    pytest.importorskip("gdstk")
    from squadds.ml.capacitance import CapacitancePredictor
    from squadds.ml.capacitance.train import trained_model

    res = CapacitancePredictor(trained_model(run), threads=1).predict(_tiny_gds(tmp_path / "t.gds"))
    assert res.names == ["G", "C1", "C2"] and np.isfinite(res.matrix_fF).all()


# ---------------------------------------------------------------------------------------------------------------- scoring
def test_layout_metric_by_hand():
    from squadds.ml.capacitance.benchmark import layout_rows

    y = np.array([[10.0, -4.0, -0.05], [-4.0, 5.0, -1.0], [-0.05, -1.0, 2.0]])
    p = y.copy()
    p[0, 1] = p[1, 0] = -4.4
    p[0, 2] = p[2, 0] = -0.06
    rows = {
        r["kind"]: r
        for r in layout_rows(y, p, np.array([False, True, True]), np.array([(0, 1), (0, 2), (1, 2)]), "a", "f")
    }
    m = rows["mutual"]
    assert m["n"] == 3
    assert m["median_pct"] == pytest.approx(10.0)  # errors 0.4/4, 0.01/max(0.05, 0.1), 0 -> 10 %, 10 %, 0 %
    assert m["mae_fF"] == pytest.approx((0.4 + 0.01) / 3)
    assert m["rmse_fF"] == pytest.approx(np.sqrt((0.16 + 1e-4) / 3))
    assert rows["signal-signal"]["n"] == 1 and rows["signal-ground"]["n"] == 2


def test_score_val_and_test_guard(run, shards, tmp_path):
    from squadds.ml.capacitance.benchmark import score

    rep = score(run, shards, split="val", threads=1)
    h = rep.headline()
    assert h["layouts"] == 6 and {"median_pct", "mae_fF", "rmse_fF"} <= set(h)
    assert rep.timing["threads"] == 1 and rep.timing["machine"]
    assert "MAE" in str(rep)
    with pytest.raises(PermissionError, match="maintainer"):
        score(run, shards, split="test")


def test_maintainer_test_scoring_once_and_disjoint(run, shards, tmp_path):
    from squadds.ml.capacitance.benchmark import score
    from squadds.ml.capacitance.training_data import read_shards, subset_shards

    test_keys = [str(k) for d in read_shards(shards) for k, s in zip(d["key"], d["split"]) if s == "val"]

    def as_test(d):
        return d["split"] == "val"

    t = subset_shards(shards, tmp_path / "t", keep=as_test)
    for p in t.glob("n*.npz"):
        d = dict(np.load(p))
        d["split"] = np.full(len(d["key"]), "test")
        np.savez_compressed(p, **d)
    with pytest.raises(ValueError, match="ledger"):
        score(run, t, split="test", maintainer=True)
    rep = score(run, t, split="test", maintainer=True, ledger_dir=tmp_path / "ledger")
    assert rep.headline()["layouts"] == len(test_keys)
    with pytest.raises(PermissionError, match="already scored"):
        score(run, t, split="test", maintainer=True, ledger_dir=tmp_path / "ledger")
    tr = subset_shards(shards, tmp_path / "tr", keep=lambda d: d["split"] == "train")
    for p in tr.glob("n*.npz"):
        d = dict(np.load(p))
        d["split"] = np.full(len(d["key"]), "test")
        np.savez_compressed(p, **d)
    with pytest.raises(PermissionError, match="training layouts"):
        score(run, tr, split="test", maintainer=True, ledger_dir=tmp_path / "ledger2")


# ---------------------------------------------------------------------------------------------------------------- workflows
def test_fit_in_family_zero_shot_wins_ties(run, tmp_path):
    from squadds.ml.capacitance.train import TrainConfig, fit_in_family

    user = make_shards(tmp_path / "user", n_layouts=8, seed=5, families=("mine",), prefix="u", val_every=99)
    res = fit_in_family(
        user, tmp_path / "cv", k_folds=4, base=run, config=TrainConfig(steps=3, log_every=3), threads=1, n_boot=200
    )
    assert set(res.report.candidate) == {"zero-shot", "from-scratch"}
    assert set(res.report.kind) >= {"mutual", "signal_self"}
    assert res.decision["folds"] == 4 and res.chosen in ("zero-shot", "from-scratch")
    if res.chosen == "zero-shot":
        assert res.model.key.startswith("base/")
    assert (Path(res.out_dir) / "decision.json").exists()


def test_retrain_collective(shards, tmp_path):
    from squadds.ml.capacitance.train import TrainConfig, retrain_collective

    user = make_shards(tmp_path / "user", n_layouts=4, seed=6, families=("mine",), prefix="u")
    model, runs = retrain_collective(
        user, shards, tmp_path / "coll", seeds=(0, 1), config=TrainConfig(steps=3, log_every=3), threads=1
    )
    assert len(runs) == 2 and len(model.weights_sha256) == 2
    import torch

    ck = torch.load(Path(model.root) / model.entry["checkpoints"][0]["path"], weights_only=False)
    assert len(ck["train_keys"]) == 28
    with pytest.raises(NotImplementedError):
        retrain_collective(user, None, tmp_path / "none")


def test_build_shards_from_gds(tmp_path):
    pytest.importorskip("gdstk")
    from squadds.ml.capacitance.training_data import Design, LabelQCError, build_shards, read_names, read_shards

    g = _tiny_gds(tmp_path / "t.gds")
    C = np.array(
        [
            [
                20.0,
                -6.0,
                -6.0,
            ],
            [-6.0, 8.0, -1.5],
            [-6.0, -1.5, 8.0],
        ]
    )
    perm = [1, 2, 0]
    d = Design(gds=str(g), labels=C[np.ix_(perm, perm)], label_names=["C1", "C2", "G"], key="tiny", family="fam")
    r = build_shards([d], tmp_path / "s", threads=1)
    assert r["layouts"] == 1 and not r["refused"]
    (s,) = read_shards(tmp_path / "s")
    assert read_names(tmp_path / "s")["tiny"] == ["G", "C1", "C2"]
    assert np.allclose(s["C"][0], C) and (s["pri"] > 0).all() and s["X"].shape == (1, 3, 61)
    bad = Design(gds=str(g), labels=C, label_names=["G", "A", "B"], key="bad")
    with pytest.raises(LabelQCError, match="never merged"):
        build_shards([bad], tmp_path / "s2", threads=1)
    r = build_shards([d, bad], tmp_path / "s3", threads=1, on_error="skip")
    assert r["layouts"] == 1 and r["refused"][0]["key"] == "bad"


def _xmon_gds(path: Path, swap_ns: bool = False) -> Path:
    """Xmon-like layout: cross S at the origin, readout claw N at the south with the (2, 0) port stub, second claw at 180
    degrees (the angle-slot namer calls it P4; the Q3D label calls it P1)."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("XMON")
    for p in gdstk.boolean(
        gdstk.rectangle((-500, -500), (500, 500)), gdstk.rectangle((-400, -400), (400, 400)), "not", layer=1
    ):
        cell.add(p)
    cross = gdstk.boolean(
        gdstk.rectangle((-150, -15), (150, 15)), gdstk.rectangle((-15, -150), (15, 150)), "or", layer=1, datatype=10
    )[0]
    north = gdstk.rectangle((-60, -230), (60, -190), layer=1, datatype=10)
    claw = gdstk.rectangle((-230, -60), (-190, 60), layer=1, datatype=10)
    for p in [cross, north, claw] if swap_ns else [north, cross, claw]:
        cell.add(p)
    cell.add(gdstk.rectangle((-5, -400), (5, -230), layer=2))
    lib.write_gds(str(path))
    return path


def _xmon_label():
    C = np.array([[30.0, -8.0, -12.0, -7.0], [-8.0, 9.0, -0.6, -0.05], [-12.0, -0.6, 13.0, -0.4]])
    C = np.vstack([C, [-7.0, -0.05, -0.4, 7.6]])
    return C, ["G", "N", "S", "P1"]


def _design_json(path: Path, polygons: list[str]) -> None:
    import json

    a = {"N": "north", "S": "south", "P1": "pad_1", "G": "ground"}
    mp = {f"signal1_{k + 1}": a[n] for k, n in enumerate(polygons + ["G"])}
    q = {"mapping_polygons": mp, "nets": {n: {"assignment": v, "net_type": "Signal"} for n, v in a.items()}}
    path.write_text(json.dumps({"ansys_gds": {"q3d_signal": q}}), encoding="utf-8")


def test_resolve_dataset_names_claw_at_180():
    pytest.importorskip("shapely")
    import shapely

    from squadds.ml.capacitance.training_data import LabelQCError, resolve_dataset_names

    north, cross = (
        shapely.box(-60, -230, 60, -190),
        shapely.box(-150, -15, 150, 15).union(shapely.box(-15, -150, 15, 150)),
    )
    claw, mk = shapely.box(-230, -60, -190, 60), {(2, 0): shapely.box(-5, -400, 5, -230)}
    lab = ["G", "N", "S", "P1"]
    assert resolve_dataset_names([north, cross, claw], mk, lab) == (["N", "S", "P1"], "dataset_file_order")
    assert resolve_dataset_names([north, cross, claw], mk, lab, ["N", "S", "P1"])[1] == "design_json"
    assert resolve_dataset_names([north, cross, claw], mk, ["G", "N", "S", "P4"]) == (["N", "S", "P4"], "dataset")
    # a polygon map that puts N on the cross fails the port-stub check; file order then fails too: refused, never merged
    with pytest.raises(LabelQCError, match="no naming"):
        resolve_dataset_names([cross, north, claw], mk, lab, ["N", "S", "P1"])
    at54, at_m18 = shapely.box(103, 150, 143, 190), shapely.box(180, -85, 220, -45)
    with pytest.raises(LabelQCError, match="ambiguous"):
        resolve_dataset_names([north, cross, at54, at_m18], mk, ["G", "N", "S", "P1", "P2"])


def test_build_shards_xmon_claw_at_180(tmp_path):
    pytest.importorskip("gdstk")
    from squadds.ml.capacitance.training_data import Design, LabelQCError, build_shards, read_names, read_shards

    C, lab = _xmon_label()
    (tmp_path / "a").mkdir()
    g = _xmon_gds(tmp_path / "a" / "x.gds")
    d = Design(gds=str(g), labels=C, label_names=lab, key="x180")
    r = build_shards([d], tmp_path / "s", threads=1)
    assert r["layouts"] == 1 and read_names(tmp_path / "s")["x180"] == ["G", "N", "S", "P1"]
    (s,) = read_shards(tmp_path / "s")
    assert np.allclose(s["C"][0], C)
    _design_json(g.with_suffix(".json"), ["N", "S", "P1"])
    build_shards([d], tmp_path / "s2", threads=1)
    import json

    assert (
        json.loads((tmp_path / "s2" / "build.json").read_text(encoding="utf-8"))["designs"]["x180"]["naming"]
        == "auto:design_json"
    )
    with pytest.raises(LabelQCError, match="never merged"):
        build_shards([Design(gds=str(g), labels=C, label_names=lab, names="dataset")], tmp_path / "s3", threads=1)
    (tmp_path / "b").mkdir()
    g2 = _xmon_gds(tmp_path / "b" / "x.gds", swap_ns=True)
    _design_json(g2.with_suffix(".json"), ["N", "S", "P1"])
    with pytest.raises(LabelQCError, match="no naming"):
        build_shards([Design(gds=str(g2), labels=C, label_names=lab)], tmp_path / "s4", threads=1)


# ---------------------------------------------------------------------------------------------------------------- slurm / CLI
def test_write_slurm_array(shards, tmp_path):
    from squadds.ml.capacitance.slurm import write_slurm_array

    p = write_slurm_array(tmp_path / "train.sh", shards, seeds=(0, 1, 2), out_dir=tmp_path / "runs", python="/opt/py")
    s = p.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2" in s and "#SBATCH --cpus-per-task=2" in s and "SEEDS=(0 1 2)" in s
    assert "--partition" not in s and "--account" not in s and "/opt/py -m squadds.ml.capacitance.train" in s
    assert "--steps 18000" in s
    s2 = write_slurm_array(
        tmp_path / "t2.sh", shards, seeds=(4,), partition="gpu", account="acct", gpus=1, device="cuda"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in s2 and "#SBATCH --account=acct" in s2 and "#SBATCH --gpus=1" in s2


def test_cli_train(shards, tmp_path, capsys):
    from squadds.ml.capacitance.train import main

    assert (
        main(["--data", str(shards), "--out", str(tmp_path / "cli"), "--seeds", "0", "--steps", "3", "--threads", "1"])
        == 0
    )
    assert (tmp_path / "cli" / "s0" / "model.pt").exists()
    assert (
        main(["--data", str(shards), "--out", str(tmp_path / "x"), "--seeds", "0,1", "--slurm", str(tmp_path / "a.sh")])
        == 0
    )
    assert "SEEDS=(0 1)" in (tmp_path / "a.sh").read_text(encoding="utf-8")
