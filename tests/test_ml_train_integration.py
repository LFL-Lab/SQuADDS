"""Data-dependent gates of the training code (tl-squadds local data; skipped when absent). Run with ``-m integration``.

(ii) reproduction of the paper seed-0 fit (VAL, never test); (iii) one in-family k-curve point; (iv) build_shards parity with
the released shards (embedding exact, priors within the macOS tolerance)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

DL = Path(os.environ.get("TLSQ_DATA_LOCAL", "/Users/shanto/LFL/tl_squadds/data_local"))
PAPER_SHARDS = DL / "learnedprior/p2d_prepared"
PAPER_S0 = DL / "geomfast/runs/p2d_full_gnn_s0/model.pt"
V2M = DL / "v4scaling/prepared_p2d_v2m"
KCURVE = DL / "v4kcurve"
FLAGSHIP_SF = DL / "flagship_v4/shards/acc_v2m_sf"
DESIGNS = DL / "designs_ood/data/design_files"


def need(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(f"data not available: {missing}")


def test_gate_ii_reproduces_paper_seed0_on_val(tmp_path):
    need(PAPER_SHARDS, PAPER_S0)
    from squadds.ml.capacitance.benchmark import paired_difference, score
    from squadds.ml.capacitance.train import fit

    # threads=1: with 2 torch threads the training step sums in a timing-dependent order, so a busy CPU changes the weights
    fit(PAPER_SHARDS, tmp_path / "s0", seed=0, threads=1, validate=False, verbose=False)
    mine = score(tmp_path / "s0", PAPER_SHARDS, split="val", threads=2)
    rel = score(PAPER_S0, PAPER_SHARDS, split="val", threads=2)
    d = paired_difference(mine, rel)
    assert mine.headline()["layouts"] == 4608
    assert abs(d["median_pct"]["diff"]) <= 0.05 and abs(d["mae_fF"]["diff"]) <= 0.01, d


def test_gate_iii_kcurve_point(tmp_path):
    run = KCURVE / "runs/xmon_k10_d0/model.pt"
    need(V2M, run, KCURVE / "manifests/xmon_k10_d0.json")
    from squadds.ml.capacitance.benchmark import paired_difference, score
    from squadds.ml.capacitance.train import fit
    from squadds.ml.capacitance.training_data import read_shards, subset_shards

    man = json.loads((KCURVE / "manifests/xmon_k10_d0.json").read_text(encoding="utf-8"))
    tr = set(man["train_keys"])
    val = [
        str(k)
        for d in read_shards(V2M)
        for k, s, f in zip(d["key"], d["split"], d["family"])
        if s == "val" and f == man["label"] and str(k) not in tr
    ]
    sub = subset_shards(V2M, tmp_path / "sub", train_keys=sorted(tr), val_keys=val)
    fit(sub, tmp_path / "run", seed=man["seed"], threads=1, validate=False, verbose=False)  # threads=1: see gate ii
    d = paired_difference(score(tmp_path / "run", sub, threads=2), score(run, sub, threads=2))
    assert abs(d["median_pct"]["diff"]) <= 0.25 and abs(d["mae_fF"]["diff"]) <= 0.05, d


def test_gate_iv_build_shards_parity(tmp_path):
    need(FLAGSHIP_SF, PAPER_SHARDS, DESIGNS)
    from squadds.ml.capacitance.training_data import Design, build_shards, read_shards

    keys = ["exp6/q3d_cap_0069", "exp9/q3d_cap_10964", "exp11/q3d_cap_1518"]
    need(*[DESIGNS / k / "cap_matrix.csv" for k in keys])
    ref = {
        str(k): {a: d[a][b] for a in ("X", "E", "G", "C", "pri")}
        for d in read_shards(FLAGSHIP_SF)
        for b, k in enumerate(d["key"])
        if str(k) in keys
    }
    ds = [
        Design(gds=str(DESIGNS / k / f"cap_{k.split('_')[-1]}.gds"), labels=str(DESIGNS / k / "cap_matrix.csv"), key=k)
        for k in keys
    ]
    build_shards(ds, tmp_path / "s", threads=2)
    for d in read_shards(tmp_path / "s"):
        for b, k in enumerate(d["key"]):
            y = ref[str(k)]
            for a in ("X", "E", "G", "C"):
                assert np.array_equal(d[a][b], y[a], equal_nan=True), (k, a)
            assert np.max(np.abs(d["pri"][b] / y["pri"] - 1)) <= 2.4e-3, k
