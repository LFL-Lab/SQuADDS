"""Retraining the planar-capacitance networks on your own compute (laptop CPU, GPU box or cluster).

- ``fit``: the paper recipe (P2d ``full_gnn``) on a shard dir, one seed. Every hyperparameter is in ``TrainConfig``.
- ``fit_seeds``: several seeds, in parallel processes (or SLURM array tasks: ``squadds.ml.capacitance.slurm``).
- ``fit_in_family``: from scratch on k of your designs, cross-validated against the zero-shot released model; zero-shot is
  always a candidate and wins ties.
- ``retrain_collective``: your designs added to the all-data training set, 3 seeds (the QCap recipe).
- ``trained_model``: a trained run (or several seeds) as a ``CapacitanceModel`` for ``CapacitancePredictor``.

The fit loop is the tl-squadds reference (``scripts/geomfast/ragged.py fit`` + QCap's NaN-label masking,
``scripts/flagship_v4/ragged_fv4.py``) statement for statement. The released fits ran in ``RELEASE_ENVIRONMENT`` (Linux
aarch64 on the GH200 node, CPU, 2 threads, its torch and numpy); a run is expected to reproduce them bit for bit only in
that environment. Other machines, thread counts, devices or library versions drift slightly (gate iii: numpy alone moves
the trajectory). The checkpoint records ``environment``, ``release_environment_match`` and the differences
(``release_environment_diff``); ``bit_identical_to_release`` is kept as an alias of ``release_environment_match``.
On CPU with ``threads > 1`` the training step sums some terms in an order that depends on thread timing, so on a busy
machine two runs of the same seed can differ in the last bits and then drift apart. ``threads=1`` (or
``torch.use_deterministic_algorithms(True)``) gives the same weights on every run on one machine.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ARMS = ("full_gnn", "direct_gnn", "full_mlp", "pair_mlp")


@dataclass(frozen=True)
class TrainConfig:
    """The paper recipe (paper_v4 Sec. 2.3-2.4). Defaults = the released models; change them knowingly."""

    steps: int = 18000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_min: float = 1e-5
    grad_clip: float = 5.0
    huber_delta: float = 0.05
    w_signal_self: float = 0.1
    w_ground_self: float = 0.02
    max_batch: int = 64
    pair_budget: int = 8192
    width: int = 64
    standardise_clip: float = 10.0
    log_every: int = 500
    # fixed in the network (squadds.ml.capacitance.network): residual bound +-0.2 in log, message-passing rounds 2 (gnn arms)


def _digest(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rounds(arm: str) -> int:
    return 2 if arm.endswith("gnn") else 0


# Environment of the released fits (paper seed-0 and QCap runs: host c02-13, Endeavour GH200, env tlsq-arm; versions
# from /project2/shaas_31/shanto/envs/tlsq-arm.freeze.txt, 2026-10-14). torch is compared without its local build tag.
RELEASE_ENVIRONMENT = {
    "machine": "aarch64",
    "system": "Linux",
    "device": "cpu",
    "threads": 2,
    "torch": "2.6.0+cu126",
    "numpy": "2.5.3",
    "host": "c02-13 (Endeavour GH200)",
    "env": "tlsq-arm",
}


def training_environment(device: str, threads: int) -> dict:
    import torch

    return {
        "machine": platform.machine().lower(),
        "system": platform.system(),
        "device": device,
        "threads": int(threads),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def release_environment_diff(device: str, threads: int, config: TrainConfig, env: dict | None = None) -> list[str]:
    """What differs from the released fits' environment and recipe (empty = same environment)."""
    env = env or training_environment(device, threads)
    rel = RELEASE_ENVIRONMENT
    diff = [
        f"{k}: {env.get(k)!r} (release {rel[k]!r})"
        for k in ("machine", "system", "device", "threads", "numpy")
        if env.get(k) != rel[k]
    ]
    if str(env.get("torch")).split("+")[0] != rel["torch"].split("+")[0]:
        diff.append(f"torch: {env.get('torch')!r} (release {rel['torch']!r})")
    if config != TrainConfig():
        diff.append("config: not the default recipe")
    return diff


def release_eligible(device: str, threads: int, config: TrainConfig, env: dict | None = None) -> bool:
    """True when the run matches the released fits' environment (``RELEASE_ENVIRONMENT``) and the default recipe: the only
    setting in which a bit-for-bit reproduction of the released checkpoints is expected."""
    return not release_environment_diff(device, threads, config, env)


def _resolve_device(device: str) -> str:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but the MPS backend is not available")
    if device not in ("cpu", "cuda", "mps") and not device.startswith("cuda:"):
        raise ValueError(f"device {device!r}: use 'cpu', 'cuda', 'cuda:<k>' or 'mps'")
    return device


def fit(
    shard_dir,
    out_dir,
    arm: str = "full_gnn",
    seed: int = 0,
    config: TrainConfig = TrainConfig(),
    device: str = "cpu",
    threads: int = 2,
    validate: bool = True,
    verbose: bool = True,
) -> dict:
    """Train one network from scratch on the ``split == "train"`` rows of a shard dir; writes ``out_dir/model.pt`` (loadable by
    ``trained_model`` / ``CapacitancePredictor``), ``run.json`` and, with ``validate``, ``metrics.csv`` / ``val_score.json`` on
    the ``val`` rows. Returns the run record."""
    import torch
    from torch.nn import functional as F

    from . import _runtime as RT
    from .network import build_model

    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    device = _resolve_device(device)
    threads = RT.configure(threads)
    cfg = config
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(shard_dir)
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    ds = [dict(np.load(p, allow_pickle=False)) for p in sorted(root.glob("n*.npz"))]
    if not ds:
        raise FileNotFoundError(f"no n*.npz shards in {root}")
    direct = arm == "direct_gnn"
    full = arm != "pair_mlp"
    rounds = _rounds(arm)
    ni = schema["nodekeys"].index("role_ground")
    ei = schema["edgekeys"].index("is_ground_pair")
    scales, centers, closure = {}, {}, {}
    train_keys = [str(k) for d in ds for k in d["key"][d["split"] == "train"]]
    if not train_keys:
        raise ValueError(f"no split == 'train' rows in {root}")
    if len(train_keys) != len(set(train_keys)):
        raise ValueError("duplicate training keys")
    for flag in [0, 1]:
        vals, bs = [], []
        for d in ds:
            tr = d["split"] == "train"
            i, j = d["pairs"].T
            mask = (d["E"][:, :, ei] == flag) & tr[:, None]
            cv = np.log((-d["C"][:, i, j][mask]).clip(1e-12))
            vals.extend(cv[np.isfinite(cv)].tolist())
            bmask = (d["sig"] == bool(flag)) & tr[:, None]
            bv = np.log(d["C"].sum(-1).clip(1e-8)[bmask])
            bs.extend(bv[np.isfinite(bv)].tolist())
        centers[str(flag)] = float(np.median(vals)) if vals else 0.0
        closure[str(flag)] = float(np.median(bs)) if bs else 0.0
    for d in ds:
        d["kind"] = d["E"][:, :, ei].copy()
        d["base"] = 2 * np.log(d["pri"][:, :, 1]) - np.log(d["pri"][:, :, 0])
        if direct:
            d["base"] = np.where(d["kind"] == 0, centers["0"], centers["1"]).astype(np.float32)
        else:
            d["E"] = np.concatenate([d["E"], np.log(d["pri"][:, :, 1:] / d["pri"][:, :, :1])], -1)
        if not full:
            d["X"] = d["X"][:, :, [ni]]
            d["G"] = np.zeros((len(d["G"]), 1), np.float32)
        d["bbase"] = np.where(d["sig"], closure["1"], closure["0"]).astype(np.float32)
    for name in ["X", "E", "G"]:
        v = np.concatenate([d[name][d["split"] == "train"].reshape(-1, d[name].shape[-1]) for d in ds])
        med = np.nanmedian(v, 0)
        iq = np.nanpercentile(v, 75, 0) - np.nanpercentile(v, 25, 0)
        iq = np.where(iq > 1e-6, iq, 1)
        scales[name] = dict(median=med.tolist(), iqr=iq.tolist())
        if not (np.isfinite(med).all() and np.isfinite(iq).all()):
            raise ValueError(f"non-finite standardisation of {name}")
        c = cfg.standardise_clip
        for d in ds:
            d[name] = np.clip((np.where(np.isfinite(d[name]), d[name], med) - med) / iq, -c, c).astype(np.float32)
    tensors, groups = [], {}
    for k, d in enumerate(ds):
        z = {n: torch.from_numpy(d[n]).to(device) for n in ["X", "E", "G", "C", "sig", "pairs", "base", "bbase"]}
        z["M"] = torch.ones(d["sig"].shape, dtype=torch.bool, device=device)
        z["PM"] = torch.ones(d["base"].shape, dtype=torch.bool, device=device)
        tensors.append(z)
        for fam in set(d["family"]):
            idx = np.flatnonzero((d["family"] == fam) & (d["split"] == "train"))
            if len(idx):
                groups.setdefault(fam, []).append((k, idx))
    families = sorted(groups)
    model = build_model(ds[0]["X"].shape[-1], ds[0]["E"].shape[-1], ds[0]["G"].shape[-1], rounds, width=cfg.width).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr_min)

    def forward(k, b):
        z = tensors[k]
        return model(
            z["X"][b], z["E"][b], z["G"][b], z["M"][b], z["PM"][b], z["pairs"], z["base"][b], z["bbase"][b], direct
        )

    hist = []
    start = time.perf_counter()
    for step in range(cfg.steps):
        fam = families[rng.integers(len(families))]
        gs = groups[fam]
        w = np.array([len(v) for k, v in gs])
        k, idx = gs[rng.choice(len(gs), p=w / w.sum())]
        z = tensors[k]
        pairs = z["pairs"]
        i, j = pairs.T
        batch = max(1, min(cfg.max_batch, cfg.pair_budget // len(pairs)))
        b = rng.choice(idx, batch)
        pred, logc = forward(k, b)
        C = z["C"][b]
        tm = (-C[:, i, j]).clamp_min(1e-12).log()
        fm = torch.isfinite(tm)
        if bool(fm.all()):
            pl = F.huber_loss(logc, tm, delta=cfg.huber_delta)
        else:
            pl = F.huber_loss(logc[fm], tm[fm], delta=cfg.huber_delta) if bool(fm.any()) else logc.new_zeros(())
        dl = F.huber_loss(
            pred.diagonal(dim1=-2, dim2=-1).log(),
            C.diagonal(dim1=-2, dim2=-1).log(),
            delta=cfg.huber_delta,
            reduction="none",
        )
        sig = z["sig"][b]
        fin = torch.isfinite(C.diagonal(dim1=-2, dim2=-1))
        if bool(fin.all()):
            loss = pl + cfg.w_signal_self * dl[sig].mean() + cfg.w_ground_self * dl[~sig].mean()
        else:
            # unknown self labels: finite placeholder targets (masked below) so no NaN reaches the gradient
            cd = C.diagonal(dim1=-2, dim2=-1)
            dl = F.huber_loss(
                pred.diagonal(dim1=-2, dim2=-1).log(),
                torch.where(fin, cd, torch.ones_like(cd)).log(),
                delta=cfg.huber_delta,
                reduction="none",
            )
            ms, mg = sig & fin, ~sig & fin
            ls = dl[ms].mean() if bool(ms.any()) else dl.new_zeros(())
            lg = dl[mg].mean() if bool(mg.any()) else dl.new_zeros(())
            loss = pl + cfg.w_signal_self * ls + cfg.w_ground_self * lg
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step} (family {fam})")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0:
            hist.append(dict(step=step, loss=float(loss.detach()), seconds=time.perf_counter() - start))
            if verbose:
                print(hist[-1], flush=True)
    train_s = time.perf_counter() - start
    from squadds import __version__

    model = model.to("cpu")
    ck = dict(
        state=model.state_dict(),
        arm=arm,
        rounds=rounds,
        scales=scales,
        schema=schema,
        closure_intercepts=closure,
        direct_intercepts=centers,
        train_keys=train_keys,
        seed=seed,
        data_sha={p.name: _digest(p) for p in root.glob("*.npz")},
        code_sha=_digest(__file__),
        steps=cfg.steps,
        config=asdict(cfg),
        machine=RT.machine(),
        device=device,
        threads=threads,
        torch_version=torch.__version__,
        package_version=f"squadds {__version__}",
        environment=training_environment(device, threads),
        release_environment=dict(RELEASE_ENVIRONMENT),
        release_environment_diff=release_environment_diff(device, threads, cfg),
        train_seconds=train_s,
    )
    ck["release_environment_match"] = not ck["release_environment_diff"]
    ck["bit_identical_to_release"] = ck["release_environment_match"]
    torch.save(ck, out / "model.pt")
    run = dict(
        arm=arm,
        seed=seed,
        steps=cfg.steps,
        data=str(root),
        host=platform.node(),
        machine=RT.machine(),
        device=device,
        threads=threads,
        train_seconds=train_s,
        environment=ck["environment"],
        release_environment_match=ck["release_environment_match"],
        release_environment_diff=ck["release_environment_diff"],
        bit_identical_to_release=ck["bit_identical_to_release"],
        config=asdict(cfg),
        history=hist,
    )
    if validate and any((d["split"] == "val").any() for d in ds):
        from .benchmark import score

        rep = score(ck, root, split="val", threads=threads, name=f"{arm} seed {seed}")
        rep.by_family.to_csv(out / "metrics.csv", index=False)
        (out / "val_score.json").write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
        run["val"] = rep.headline()
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    if verbose:
        print("DONE", arm, seed, f"{train_s:.1f} s", run.get("val", ""), flush=True)
    return run


def _fit_job(kw):
    return fit(**kw)


def fit_seeds(
    shard_dir,
    out_dir,
    seeds=(0, 1, 2),
    processes: int = 1,
    arm: str = "full_gnn",
    config: TrainConfig = TrainConfig(),
    device: str = "cpu",
    threads: int = 2,
    validate: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """``fit`` for each seed into ``out_dir/s<seed>``; ``processes`` seeds at a time (spawned processes, ``threads`` each)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        dict(
            shard_dir=str(shard_dir),
            out_dir=str(out / f"s{s}"),
            arm=arm,
            seed=int(s),
            config=config,
            device=device,
            threads=threads,
            validate=validate,
            verbose=verbose,
        )
        for s in seeds
    ]
    if processes <= 1 or len(jobs) == 1:
        return [fit(**j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(min(processes, len(jobs)), mp_context=get_context("spawn")) as ex:
        return list(ex.map(_fit_job, jobs))


def trained_model(runs, name: str = "user", title: str | None = None):
    """A ``CapacitanceModel`` (ensemble mean over the given runs) for ``CapacitancePredictor``: run dirs or ``model.pt`` paths.
    Prior-based arms run on the corrected prior stack (the one the released models serve); the embedding is exact."""
    import torch

    from .benchmark import checkpoint_paths
    from .models import CapacitanceModel, sha256_file

    paths = [p.resolve() for p in checkpoint_paths(list(runs) if isinstance(runs, (list, tuple)) else runs)]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    arms = {c["arm"] for c in cks}
    if len(arms) != 1:
        raise ValueError(f"runs mix arms {arms}")
    arm = arms.pop()
    root = Path(os.path.commonpath([str(p.parent) for p in paths]))
    shas = [sha256_file(p) for p in paths]
    key = f"{name}-{hashlib.sha256(''.join(shas).encode()).hexdigest()[:8]}"
    entry = {
        "title": title or f"{name}: {arm} x{len(paths)} (user-trained)",
        "kind": "gnn",
        "arm": arm,
        "stack": None if arm == "direct_gnn" else "corrected",
        "checkpoints": [
            {"name": p.parent.name, "path": str(p.relative_to(root)), "sha256": h} for p, h in zip(paths, shas)
        ],
        "trained": [
            {
                k: c.get(k)
                for k in (
                    "seed",
                    "steps",
                    "machine",
                    "device",
                    "threads",
                    "release_environment_match",
                    "bit_identical_to_release",
                )
            }
            for c in cks
        ],
    }
    if arm != "direct_gnn":
        entry["ppg"] = [1.5, 3.0]
    return CapacitanceModel(key, entry, root, "local", None, shas)


# ---------------------------------------------------------------------------------------------------------------- in family
def _base_model(base):
    from .models import CapacitanceModel, is_model_name, load_model

    if isinstance(base, CapacitanceModel):
        return base
    if is_model_name(base):
        return load_model(base)
    return trained_model(base, name="base")


def _folds(keys: list[str], k_folds: int, seed: int) -> list[list[str]]:
    keys = sorted(keys)
    perm = np.random.default_rng(seed).permutation(len(keys))
    k = max(2, min(int(k_folds), len(keys)))
    return [[keys[i] for i in perm[f::k]] for f in range(k)]


@dataclass
class InFamilyResult:
    chosen: str
    model: object
    report: object
    decision: dict
    out_dir: str
    timing: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"in-family CV ({self.decision['layouts']} designs): chosen = {self.chosen}; {self.decision['rule']}"


def fit_in_family(
    designs,
    out_dir,
    k_folds: int = 5,
    base="QCap",
    seed: int = 0,
    config: TrainConfig = TrainConfig(),
    device: str = "cpu",
    threads: int = 2,
    processes: int = 1,
    final_seeds=(0,),
    margin_pp: float = 0.25,
    margin_fF: float = 0.05,
    n_boot: int = 2000,
    family: str = "user",
    build_kwargs: dict | None = None,
) -> InFamilyResult:
    """From-scratch training on your k designs, cross-validated against the zero-shot ``base`` model.

    designs: Design objects / dicts (built here with ``build_shards``) or an existing shard dir (every row used). k-fold CV
    (deterministic folds from ``seed``): each fold is predicted by a network trained from scratch on the other folds, and by
    the zero-shot base on the same stored embedding and priors. The from-scratch recipe is chosen only if it is materially
    better (paired layout-bootstrap 95% interval of the layout-mean difference excludes zero AND the gain exceeds ``margin_pp``
    percentage points of median error or ``margin_fF`` of MAE: the paper's equivalence rule); otherwise the zero-shot model
    is kept (it wins ties). The chosen from-scratch model is then trained on all k designs (``final_seeds``).
    Reported per entry kind: layout-mean median %, MAE and RMSE (fF), with time, machine and threads."""
    import pandas as pd

    from .benchmark import layout_rows, load_checkpoints, paired_difference, predict_arrays, summarize
    from .training_data import build_shards, read_schema, read_shards, subset_shards

    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    if isinstance(designs, (str, Path)) and Path(designs).is_dir():
        data = subset_shards(designs, out / "data", keep=lambda d: np.ones(len(d["key"]), bool))
    else:
        from .training_data import as_design

        ds = [dataclasses.replace(as_design(x), family=family, split="train") for x in designs]
        build_shards(ds, out / "data", threads=threads, **(build_kwargs or {}))
        data = out / "data"
    t_build = time.perf_counter() - t0
    schema = read_schema(data)
    shards = read_shards(data)
    keys = [str(k) for d in shards for k in d["key"]]
    if len(keys) < 2:
        raise ValueError("in-family CV needs at least 2 designs")
    folds = _folds(keys, k_folds, seed)
    jobs = []
    for f, val in enumerate(folds):
        sub = subset_shards(
            data, out / "folds" / f"f{f}" / "data", train_keys=[k for k in keys if k not in set(val)], val_keys=val
        )
        jobs.append(
            dict(
                shard_dir=str(sub),
                out_dir=str(out / "folds" / f"f{f}" / "run"),
                arm="full_gnn",
                seed=seed,
                config=config,
                device=device,
                threads=threads,
                validate=False,
                verbose=False,
            )
        )
    t1 = time.perf_counter()
    if processes > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with ProcessPoolExecutor(min(processes, len(jobs)), mp_context=get_context("spawn")) as ex:
            runs = list(ex.map(_fit_job, jobs))
    else:
        runs = [fit(**j) for j in jobs]
    t_cv = time.perf_counter() - t1
    bm = _base_model(base)
    zs = load_checkpoints(bm, threads)
    rows = {"zero-shot": [], "from-scratch": []}
    for f, val in enumerate(folds):
        fm = load_checkpoints(Path(jobs[f]["out_dir"]), threads)
        for d in shards:
            ix = np.flatnonzero(np.isin(d["key"].astype(str), val))
            if not len(ix):
                continue
            for arm, mods in (("zero-shot", zs), ("from-scratch", fm)):
                P = predict_arrays(mods, d, schema, ix)
                for b, p in zip(ix, P):
                    rows[arm] += layout_rows(
                        d["C"][b].astype(float), p, d["sig"][b], d["pairs"], str(d["key"][b]), family
                    )
    L = {a: pd.DataFrame(r) for a, r in rows.items()}
    tabs = []
    for a, frame in L.items():
        _, tot = summarize(frame)
        tot.insert(0, "candidate", a)
        tabs.append(tot)
    report = pd.concat(tabs, ignore_index=True)
    diff = paired_difference(L["from-scratch"], L["zero-shot"], n_boot=n_boot, seed=seed)
    dm, dmae = diff["median_pct"], diff["mae_fF"]
    better = (dm["hi"] < 0 and -dm["diff"] > margin_pp) or (dmae["hi"] < 0 and -dmae["diff"] > margin_fF)
    chosen = "from-scratch" if better else "zero-shot"
    decision = dict(
        layouts=len(keys),
        folds=len(folds),
        paired_from_scratch_minus_zero_shot=diff,
        margin_pp=margin_pp,
        margin_fF=margin_fF,
        rule="from-scratch only if materially better (95% interval excludes zero and the gain exceeds the margin); "
        "zero-shot wins ties",
        chosen=chosen,
        base=bm.name,
    )
    t2 = time.perf_counter()
    if chosen == "from-scratch":
        fit_seeds(
            data,
            out / "final",
            seeds=final_seeds,
            processes=processes,
            config=config,
            device=device,
            threads=threads,
            validate=False,
        )
        model = trained_model([out / "final" / f"s{s}" for s in final_seeds], name=f"{family}-in-family")
    else:
        model = bm
    from ._runtime import machine

    timing = dict(
        build_s=t_build,
        cv_fits_s=t_cv,
        per_fit_s=[r["train_seconds"] for r in runs],
        final_s=time.perf_counter() - t2,
        total_s=time.perf_counter() - t0,
        threads=threads,
        processes=processes,
        device=device,
        machine=machine(),
    )
    report.to_csv(out / "cv_report.csv", index=False)
    pd.concat([f.assign(candidate=a) for a, f in L.items()]).to_csv(out / "cv_layouts.csv", index=False)
    (out / "decision.json").write_text(
        json.dumps(dict(decision=decision, timing=timing), indent=1, default=str), encoding="utf-8"
    )
    return InFamilyResult(chosen, model, report, decision, str(out), timing)


def retrain_collective(
    user_shards,
    base_shards,
    out_dir,
    seeds=(0, 1, 2),
    processes: int = 1,
    config: TrainConfig = TrainConfig(),
    device: str = "cpu",
    threads: int = 2,
):
    """Your shards added to the all-data training set (``base_shards``: the QCap shard dir, i.e. the dataset's
    ``model_input`` on the corrected prior), every row trained on, ``seeds`` networks from scratch (QCap recipe).
    Returns (CapacitanceModel of the ensemble, run records)."""
    from .training_data import merge_shards

    if base_shards is None:
        raise NotImplementedError(
            "base_shards=None: download the dataset's model_input table and convert it with "
            "training_data.shards_from_model_input (the release download is not wired yet)"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    data = merge_shards(base_shards, user_shards, out / "data", all_split="train")
    runs = fit_seeds(
        data,
        out / "runs",
        seeds=seeds,
        processes=processes,
        config=config,
        device=device,
        threads=threads,
        validate=False,
    )
    return trained_model([out / "runs" / f"s{s}" for s in seeds], name="collective"), runs


# ---------------------------------------------------------------------------------------------------------------- CLI
def add_train_subcommand(sub) -> None:
    """Register ``squadds-ml train`` on an argparse subparsers object (for ``cli.py``)."""
    p = sub.add_parser("train", help="retrain the capacitance networks on a shard dir (paper recipe)")
    p.add_argument("--data", required=True, help="shard dir (n0NN.npz + schema.json)")
    p.add_argument("--out", required=True, help="output dir (one sub-dir per seed)")
    p.add_argument("--arm", default="full_gnn", choices=ARMS)
    p.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2")
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--slurm", default=None, help="write a SLURM array script here instead of training")
    p.add_argument("--partition", default=None)
    p.add_argument("--account", default=None)
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--seed-index", type=int, default=None, help="train only seeds[k] (SLURM array task)")
    p.set_defaults(func=cmd_train)


def cmd_train(a) -> int:
    seeds = [int(s) for s in str(a.seeds).split(",") if s != ""]
    if a.slurm:
        from .slurm import write_slurm_array

        path = write_slurm_array(
            a.slurm,
            a.data,
            seeds,
            out_dir=a.out,
            partition=a.partition,
            account=a.account,
            cpus=a.threads,
            time=a.time,
            arm=a.arm,
            steps=a.steps,
            device=a.device,
        )
        print(f"wrote {path}: sbatch {path}")
        return 0
    if a.seed_index is not None:
        seeds = [seeds[a.seed_index]]
    cfg = TrainConfig(steps=a.steps)
    runs = fit_seeds(
        a.data,
        a.out,
        seeds=seeds,
        processes=a.processes,
        arm=a.arm,
        config=cfg,
        device=a.device,
        threads=a.threads,
        validate=not a.no_validate,
        verbose=a.processes <= 1,
    )
    for r in runs:
        print(
            json.dumps(
                {k: r[k] for k in ("arm", "seed", "steps", "train_seconds", "machine", "threads", "device") if k in r}
                | ({"val": r["val"]} if "val" in r else {})
            )
        )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m squadds.ml.capacitance.train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_train_subcommand(sub)
    a = ap.parse_args(["train", *(argv if argv is not None else sys.argv[1:])])
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
