"""Model arms: the shared inference stage of the released models (frozen copy, provenance in SOURCES.json).

Every arm runs from a layout file on a CPU, through one code path:
scene (geomgnn.api.scene_from_prepared) -> geometric embedding -> priors at the arm's mesh levels under the arm's prior stack
-> the arm's combination. Kinds:
  gnn       geomfast networks (infer.Predictor): full_gnn / direct_gnn / full_mlp / pair_mlp checkpoints; the two levels are
            fed as the networks' (1.5, 3) inputs (the coarser-pair arm reads R(0.75, 1.5) there, as it was trained);
            learned_prior: the fast model's surrogate replaces the solves
  multifid  cheap-physics networks (x1_core.Multifid, scripts/multifid/train_mf.py); learned-exponent arms (curve) use the
            train_mf.fit anchor (finest level) and curve=True
  physics   no learned parameter: one level (to_maxwell) or a Richardson pair (x1_core.c0)
  final     the prior-generation model and its fast mode (tlsq_infer.final, unchanged)
Prior stacks (paper_hub.prior_stacks) are process-wide (geomgnn.api.set_prior_stack); each request installs its arm's stack.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import __version__
from . import api as A
from . import infer as I
from . import x1_core as X
from ._tlsq.embed import fastprior, proxy
from ._tlsq.uge.graph import scene_graph

lt = I.lt
ALPHA = 0.49685219816506904  # TRAIN-fitted gap share of every paper prior (P2d physics.alpha)
_CACHE: dict = {}


def arms(reg) -> dict:
    return reg.data.get("paper_arms") or {}


def live(reg) -> list[str]:
    return [k for k, v in arms(reg).items() if v.get("status", "available") == "available"]


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _model_dir(model_dir=None):
    return os.environ.get("TLSQ_MODEL_DIR") if model_dir is None else model_dir


def files(reg, name, key="checkpoints", model_dir=None) -> list[Path]:
    """Bundle copies (<model_dir>/<name>/info.json) or the registry's source paths."""
    a = arms(reg)[name]
    md = _model_dir(model_dir)
    info = Path(md) / name / "info.json" if md else None
    if info is not None and info.exists():
        return [Path(md) / name / f for f in json.loads(info.read_text()).get(key, [])]
    src = a.get(key) or []
    if key == "surrogate":
        src = [a["learned_prior"]["checkpoint"]] if a.get("learned_prior") else []
    return [reg.root / c["path"] for c in src]


def _hashes(a, key):
    if key == "surrogate":
        return [a["learned_prior"]["checkpoint"]["sha256"]] if a.get("learned_prior") else []
    return [c["sha256"] for c in a.get(key) or []]


def available(name, reg, model_dir=None) -> bool:
    a = arms(reg)[name]
    if a["kind"] == "final":
        from tlsq_infer import final as F
        return F.available(a["final_model"], reg)
    return all(f.exists() for k in ("checkpoints", "surrogate") for f in files(reg, name, k, model_dir))


def stack_of(reg, a) -> dict | None:
    s = a.get("stack")
    return None if s is None else reg["paper_hub"]["prior_stacks"][s]


def _check(reg, name, model_dir=None):
    a = arms(reg)[name]
    for key in ("checkpoints", "surrogate"):
        fs, hs = files(reg, name, key, model_dir), _hashes(a, key)
        assert len(fs) == len(hs), (name, key)
        for f, h in zip(fs, hs):
            if _sha(f) != h:
                raise RuntimeError(f"{name}: {f.name} SHA256 differs from the registry ({h[:12]})")
    return a


class _Curve:
    """Learned-exponent cheap-physics arms (train_mf.fit with curve=True): anchor = log of the finest level, slope = finest -
    next level, the head's exponent p sets m = anchor + slope / (2^p - 1); otherwise x1_core.Multifid.predict_arrays."""

    def __init__(self, paths, threads=2):
        import torch

        from .train_mf import build
        torch.set_num_threads(threads)
        self.models = []
        for p in paths:
            c = torch.load(p, map_location="cpu", weights_only=False)
            assert c["curve"]
            m = build(len(c["scales"]["X"]["median"]), len(c["scales"]["E"]["median"]), len(c["scales"]["G"]["median"]), 2)
            m.load_state_dict(c["state"])
            m.eval()
            self.models.append((c, m))
        self.levels = self.models[0][0]["levels"]
        assert all(c["levels"] == self.levels for c, _ in self.models)

    def predict(self, row, levels):
        nk, ek, gk = X.schema_of(self.models[0][0])
        X0, E0, G0, pairs = X.raw_arrays(row, nk, ek, gk)
        i, j = pairs.T
        sig = np.array([nm != "G" for nm in row["names"]])
        lev = {p: np.asarray(levels[p], float)[i, j] for p in self.levels}
        import torch
        out = []
        for c, m in self.models:
            L = [np.log(np.asarray(lev[p], float)) for p in self.levels]
            assert all(np.isfinite(v).all() for v in L)
            lf, lc = L[-1], L[-2]
            anchor, slope = lf, lf - lc
            E = np.concatenate([E0] + [(L[k + 1] - L[k])[:, None].astype(np.float32) for k in range(len(L) - 1)], -1)
            Xn, En, Gn = X.norm(X0, c["scales"]["X"])[None], X.norm(E, c["scales"]["E"])[None], X.norm(G0[None], c["scales"]["G"])
            bbase = np.where(sig, c["closure"]["1"], c["closure"]["0"]).astype(np.float32)[None]
            T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
            n = len(sig)
            with torch.inference_mode():
                C = m(T(Xn), T(En), T(Gn), torch.ones((1, n), dtype=torch.bool), torch.ones((1, len(pairs)), dtype=torch.bool),
                      torch.from_numpy(pairs), T(anchor[None].astype(np.float32)), T(slope[None].astype(np.float32)), T(bbase), True)[0]
            out.append(C[0].numpy().astype(np.float64))
        r = np.mean(out, 0)
        assert np.isfinite(r).all()
        return r


def load(name, reg, model_dir=None, threads=2):
    """The arm's network object (cached): infer.Predictor (gnn, with .surrogate), x1_core.Multifid / _Curve (multifid), None."""
    key = (str(reg.path), name, str(_model_dir(model_dir)), int(threads))
    if key in _CACHE:
        return _CACHE[key]
    a = _check(reg, name, model_dir)
    obj = None
    if a["kind"] == "gnn":
        obj = I.Predictor(files(reg, name, "checkpoints", model_dir), threads=threads)
        if obj.arm != a["arm"]:
            raise RuntimeError(f"{name}: checkpoints are arm {obj.arm!r}, registry says {a['arm']!r}")
        phys = obj.models[0][0].get("schema", {}).get("physics", {})
        if not obj.direct and float(phys.get("alpha")) != ALPHA:
            raise RuntimeError(f"{name}: checkpoint alpha {phys.get('alpha')} != {ALPHA}")
        obj.surrogate = None
        if a.get("learned_prior"):
            from . import learned_prior
            model, prep, _ = learned_prior.load(files(reg, name, "surrogate", model_dir)[0])
            prep.floor_ff = a["learned_prior"].get("floor_ff")
            obj.surrogate = (model, prep)
        if obj.direct != (a.get("stack") is None and not a.get("learned_prior")):
            raise RuntimeError(f"{name}: solver-free checkpoints iff no prior stack and no learned prior")
    elif a["kind"] == "multifid":
        paths = files(reg, name, "checkpoints", model_dir)
        import torch
        curve = torch.load(paths[0], map_location="cpu", weights_only=False)["curve"]
        obj = _Curve(paths, threads) if curve else X.Multifid(paths, threads)
        if [float(p) for p in obj.levels] != [float(p) for p in a["levels"]]:
            raise RuntimeError(f"{name}: checkpoint levels {obj.levels} != registry {a['levels']}")
    _CACHE[key] = obj
    return obj


def levels_of(a) -> list[float]:
    return [float(p) for p in (a.get("levels") or a.get("ppg") or [])]


def apply_stack(reg, a, threads=2):
    st = stack_of(reg, a)
    if st is not None:
        A.set_prior_impl("frozen")
        A.set_prior_stack(st, threads)
        A._thread_limits(threads)


_LEVELS: OrderedDict = OrderedDict()  # solved prior levels: (layout, level, substrate, stack) -> P; TLSQ_PRIOR_LEVEL_CACHE_MB (0 = off)
_MESHES: OrderedDict = OrderedDict()  # the prior's window meshes (geometry and lattice only, no substrate), same bound


def _geom_hash(geoms) -> str:
    import shapely
    h = hashlib.sha256()
    for g in geoms:
        h.update(shapely.to_wkb(g))
        h.update(b"|")
    return h.hexdigest()


def _level_key(sc, threads, eps_r, t_um) -> tuple:
    return (_geom_hash([c.geom for c in sc.conductors]), float(eps_r), float(t_um), int(threads), ALPHA, A._STATE["prior_impl"],
            repr(A._STATE.get("stack")), repr(A._STATE.get("pr")))


def _put(cache, k, v, cap) -> None:
    cache[k] = v
    nb = lambda x: sum(a.nbytes for a in (x if isinstance(x, tuple) else (x,)) if isinstance(a, np.ndarray))  # noqa: E731
    while len(cache) > 1 and sum(nb(x) for x in cache.values()) > cap:
        cache.popitem(last=False)


def _memo_mesh(f, cap):
    """f(metals, *args) -> mesh arrays, memoised on the metals' WKB and the arguments; hits return copies."""
    def g(metals, *args, **kw):
        k = (f.__module__, f.__name__, _geom_hash(metals), repr(args), repr(sorted(kw.items())))
        v = _MESHES.get(k)
        if v is None:
            v = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in f(metals, *args, **kw))
            _put(_MESHES, k, v, cap)
        else:
            _MESHES.move_to_end(k)
        return tuple(x.copy() if isinstance(x, np.ndarray) else x for x in v)
    return g


def solve_levels(sc, levels, threads=2, stack=None) -> tuple[dict, dict]:
    """Positive-mutual prior matrices P = -C at each level under the installed stack (the statements of
    geomgnn.api.predict_scene's prior loop, any level list); stack = (eps_r, thickness_um) of the substrate the prior is solved at
    (default the scene's, i.e. the training stack). With TLSQ_PRIOR_LEVEL_CACHE_MB > 0 every solved level is kept (LRU, bounded
    in MB) and reused by any later call with the same layout, level, substrate and prior stack (tiers sharing a level), and the
    window meshes are reused across substrates (same layout and level)."""
    eps_r, t_um = stack or (sc.stack["substrate_eps_r"], sc.stack["substrate_thickness_um"])
    cap = float(os.environ.get("TLSQ_PRIOR_LEVEL_CACHE_MB", "0")) * 2**20
    key = _level_key(sc, threads, eps_r, t_um) if cap > 0 and levels else None
    out, tm = {}, {}
    for p in levels if key else ():
        v = _LEVELS.get(key + (float(p),))
        if v is not None:
            _LEVELS.move_to_end(key + (float(p),))
            out[p], tm[f"{p:g}"] = v.copy(), 0.0
    todo = [p for p in levels if p not in out]
    if todo:
        from ._tlsq.embed import fastprior_loc as FL
        from . import priorv2
        hooks = [(m, n, getattr(m, n)) for m, n in ((priorv2, "_mesh"), (FL, "mesh_local"))] if key else []
        try:
            for m, n, f in hooks:
                setattr(m, n, _memo_mesh(f, cap))
            P, t = _solve_levels(sc, todo, threads, eps_r, t_um)
        finally:
            for m, n, f in hooks:
                setattr(m, n, f)
        out.update(P)
        tm.update(t)
        for p in todo if key else ():
            _put(_LEVELS, key + (float(p),), P[p].copy(), cap)
    return {p: out[p] for p in levels}, {f"{p:g}": tm[f"{p:g}"] for p in levels}


def _solve_levels(sc, levels, threads, eps_r, t_um) -> tuple[dict, dict]:
    from ._tlsq.embed.fastprior_loc import micro_mom_loc
    from . import levelpar
    if levelpar.enabled(sc, levels, threads):  # the levels concurrently (spawned workers, same stack and statements)
        return levelpar.solve(sc, list(levels), threads, ALPHA, (eps_r, t_um))
    proxy.NUM_THREADS = threads
    if set(A.V2_KWARGS) & set(A._STATE["patches"]):
        from . import priorv2
        priorv2.clear()
    out, tm = {}, {}
    old = fastprior.ALPHA_GAP_SHARE
    try:
        fastprior.ALPHA_GAP_SHARE = ALPHA
        for p in levels:
            t = time.perf_counter()
            with proxy.substrate("slab_open", t_um), A._quiet_mixed_cast():
                mat, _, _ = micro_mom_loc(sc.conductors[0].geom, [c.geom for c in sc.conductors[1:]], eps_r=eps_r, ppg=p)
            out[p] = -mat
            tm[f"{p:g}"] = time.perf_counter() - t
    finally:
        fastprior.ALPHA_GAP_SHARE = old
    return out, tm


def combine(a, obj, row, P) -> np.ndarray:
    """The arm's Maxwell matrix from the embedding row and the priors P {ppg: -C} (scene order)."""
    k = a["kind"]
    if k == "physics":
        lv = levels_of(a)
        return X.to_maxwell(P[lv[0]]) if len(lv) == 1 else X.c0(P, lv[0], lv[1])
    if k == "multifid":
        return obj.predict(row, P)
    if obj.direct:
        return obj.from_features(row)
    if obj.surrogate is not None:
        from . import learned_prior
        return obj.from_features(row, learned_prior.priors(*obj.surrogate, row))
    lo, hi = levels_of(a)
    return obj.from_features(row, {1.5: P[lo], 3.0: P[hi]})


def predict_scene(reg, name, sc, threads=2, model_dir=None, stack=None) -> dict:
    """The arm's prediction; an arm's `embedding: {far_k, d_far_um}` (flagship tiers only, agents/embedspeed Part A) selects the
    far-sample decimation of the pair vectors for this call (the paper's arms always use the exact embedding)."""
    from ._tlsq.uge import blocks as _B
    emb = {} if os.environ.get("TLSQ_EMBEDDING_EXACT") == "1" else (arms(reg)[name].get("embedding") or {})  # tests of the exact recipe
    old = dict(_B.FAR_DECIMATION)
    _B.FAR_DECIMATION.update(k=int(emb.get("far_k", 0)), d_far=float(emb.get("d_far_um", 100.0)))
    try:
        return _predict_scene(reg, name, sc, threads, model_dir, stack)
    finally:
        _B.FAR_DECIMATION.clear()
        _B.FAR_DECIMATION.update(old)


def _predict_scene(reg, name, sc, threads=2, model_dir=None, stack=None) -> dict:
    a = arms(reg)[name]
    obj = load(name, reg, model_dir, threads)
    apply_stack(reg, a, threads)
    t0 = time.perf_counter()
    need = [] if (a["kind"] == "gnn" and (obj.direct or obj.surrogate is not None)) else levels_of(a)
    from . import overlap as OV
    fut = OV.start(sc, threads) if need and OV.enabled(sc, threads) else None  # the embedding runs in the worker during the priors
    row = None if fut else lt.add_tokens(sc, scene_graph(sc))
    t1 = time.perf_counter()
    P, tl = solve_levels(sc, need, threads, stack) if need else ({}, {})
    t2 = time.perf_counter()
    extra = {}
    if fut is not None:
        row, embed_s = fut.result()
        extra = dict(overlap=True, geometry_worker=embed_s, geometry_wait_after_prior=time.perf_counter() - t2)
        t2 = time.perf_counter()
    C = np.asarray(combine(a, obj, row, P), float)
    t3 = time.perf_counter()
    return dict(C=C, names=row["names"], row=row, priors=P,
                timing=dict(geometry=t1 - t0, prior=t2 - t1, prior_levels=tl, network=t3 - t2, total=t3 - t0, **extra))


def route(a, eps, t_um) -> tuple[dict, list]:
    """Substrate route of an arm (geomgnn.api.substrate_route): arms that solve a prior solve it at (eps, t_um); learned-prior arms (the
    Fast tier) are eps-rescaled off the stack with a warning; the solver-free arm has neither and serves the training stack only."""
    if a["kind"] == "gnn" and a.get("stack") is None and not a.get("learned_prior"):
        if not A.on_training_stack(eps, t_um):
            raise ValueError(f"the solver-free model has no physics prior to solve at another stack: it is served for Si (eps_r "
                             f"{A.EPS_TRAIN:g}), {A.T_TRAIN_UM:g} um only (requested eps_r {eps:g}, {t_um:g} um)")
        return A.substrate_route(eps, t_um, solves_prior=False)
    sub, w = A.substrate_route(eps, t_um, solves_prior=not a.get("learned_prior"))
    if a["kind"] == "physics" and sub.get("caveat"):
        sub["caveat"] = "physics prior alone (no network), solved at your stack"
    return sub, w


def predict(prepared, name, reg, substrate="Si (cryogenic, training)", thickness_um=350.0, model_dir=None, threads=None,
            self_from_rows=True, trust_rule=None) -> dict:
    """Full N x N Maxwell matrix (fF; ground row/column 0) of a prepared layout with a paper_arms entry."""
    a = arms(reg)[name]
    threads = int(threads or os.environ.get("TLSQ_NUM_THREADS") or 2)
    if a["kind"] == "final":
        from tlsq_infer import final as F
        tr = bool(reg["final_defaults"].get("trust_rule", True)) if trust_rule is None else bool(trust_rule)
        res = F.predict_final(prepared, a["final_model"], substrate=substrate, thickness_um=thickness_um, self_from_rows=bool(self_from_rows),
                              trust_rule=tr if a.get("trust_rule_applies") else False)
        res.update(arm=name, arm_title=a["title"], paper_name=a["paper_name"], route=a.get("route"))
        return res
    eps, t_um = reg.eps_of(substrate), float(thickness_um)
    sub, sub_w = route(a, eps, t_um)
    stack = None if A.on_training_stack(eps, t_um) else (eps, t_um)
    t = time.perf_counter()
    load(name, reg, model_dir, threads)
    load_s = time.perf_counter() - t
    sc = A.scene_from_prepared(prepared)
    try:
        r = predict_scene(reg, name, sc, threads, model_dir, stack)
    except AssertionError as e:
        raise ValueError(f"{name}: the prior or the features of this layout are not finite (no matrix returned): {e}") from e
    C = r["C"] if stack is None or sub["eps_rescale_factor"] is None else r["C"] * sub["eps_rescale_factor"]
    chk = A.physical_checks(C)
    warnings = [w for w in prepared.warnings if "beyond the validated range" not in w]
    if a.get("warning"):
        warnings.insert(0, a["warning"])
    pw = A.physical_warning(chk, C)
    if pw:
        warnings.insert(0, pw)
    warnings += sub_w
    st = stack_of(reg, a)
    tm = r["timing"]
    return {"names": ["G"] + list(prepared.names), "matrix_fF": C.tolist(), "units": "fF",
            "convention": "Maxwell matrix: diagonal = self, off-diagonal = -mutual; row/column 0 = ground",
            "self_capacitance": a.get("self_note") or ("predicted by the network (learned positive row-sum closure). The ground self term "
                                                       "depends on the chip boundary: indicative only"),
            "model": name, "arm": name, "arm_title": a["title"], "paper_name": a["paper_name"], "route": a.get("route"), "kind": a["kind"],
            "network_arm": a.get("arm"), "levels_ppg": levels_of(a) if a["kind"] != "gnn" or not (a.get("learned_prior") or a.get("stack") is None) else [],
            "weights_sha256": _hashes(a, "checkpoints"), "learned_prior_sha256": (_hashes(a, "surrogate") or [None])[0],
            "prior_stack": (None if st is None else {"name": a["stack"], "patches": A._STATE.get("patches", []), "mode": A._STATE.get("mode") or "frozen",
                                                     "alpha": ALPHA, "method": "local_gap micro-MoM (slab)"}),
            "package_version": f"geomgnn {__version__}", "registry_version": reg["version"],
            "substrate": substrate if not isinstance(substrate, (int, float)) else f"custom eps={eps:g}", "eps_r": eps, "thickness_um": t_um,
            "substrate_route": sub, "training_stack": {"eps_r": A.EPS_TRAIN, "thickness_um": A.T_TRAIN_UM},
            "timing_s": {"load_s": load_s, "geometry_s": tm["geometry"], "prior_s": tm["prior"], "prior_levels_s": tm["prior_levels"],
                         "network_s": tm["network"], "total_s": tm["total"], "threads": threads},
            "paper": a.get("paper") or {}, "checks": chk, "warnings": warnings, "stored": False}


STARTUP: dict = {}


def preload(reg, example=None, names=None) -> dict:
    """Server start: load every live arm (SHA checks) and run the default arm on a small example (numba warmup of the compiled
    solver included), so the first request does not pay the compile."""
    t0 = time.perf_counter()
    for k in names or live(reg):
        if arms(reg)[k]["kind"] != "final" and available(k, reg):
            load(k, reg)
    STARTUP["load_all_s"] = time.perf_counter() - t0
    d = reg["paper_hub"]["default_arm"]
    if example is not None and Path(example).exists():
        from .layout import prepare
        t = time.perf_counter()
        predict(prepare(str(example)), d, reg)
        STARTUP["example_s"] = time.perf_counter() - t
        if os.environ.get("TLSQ_PRELOAD_ALL", "1") != "0":  # every live arm once (each stack's compile, the surrogates)
            t = time.perf_counter()
            for k in names or live(reg):
                if k != d and arms(reg)[k]["kind"] != "final" and available(k, reg):
                    predict(prepare(str(example)), k, reg)
            STARTUP["example_all_arms_s"] = time.perf_counter() - t
    t = time.perf_counter()
    from ._tlsq.uge import nnk
    STARTUP["nn_kernel"] = nnk.available() and nnk._STATE.get("arith")  # probe + compile of the exact NN kernel
    from . import overlap
    th = int(os.environ.get("TLSQ_NUM_THREADS") or 2)
    if th >= int(os.environ.get("TLSQ_OVERLAP_MIN_THREADS", "4")):
        overlap._executor(th)  # the embedding worker of large layouts, spawned now
    STARTUP["kernel_and_worker_s"] = time.perf_counter() - t
    STARTUP.update(default=d, numba_warmup_s=A._STATE.get("warmup_s"), total_s=time.perf_counter() - t0)
    print("preload " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in STARTUP.items()}), flush=True)
    return dict(STARTUP)
