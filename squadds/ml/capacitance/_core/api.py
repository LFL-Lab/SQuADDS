"""Registry-driven GNN prediction for the Space (deploy/registry.yaml: gnn_hub, gnn_models).

predict_gnn(prepared, name) runs the frozen Predictor (geomgnn.infer, FINALIZATION_V1 checkpoints) on the scene built from a
tlsq_infer PreparedLayout. The steps after the scene are those of infer.Predictor.from_gds, statement for statement;
deploy/tests/test_gnn_space.py checks the result against the frozen stage's from_gds on the raw TRAIN GDS files.
Entries with `learned_prior` (the fast tier) replace the two field solves by the learned prior surrogate
(geomgnn.learned_prior = scripts/learnedprior/lp_bench.py Learned.from_gds); the networks are the same frozen weights.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import shapely

from . import __version__
from . import infer as I
from ._tlsq.embed import fastprior, proxy
from ._tlsq.uge.graph import scene_graph
from ._tlsq.uge.scene import DEFAULT_STACK, Conductor, Scene
from ._tlsq.uge import blocks as _blocks

# Inference default: the numerically exact embedding (translated copies shared; last-bit differences only, agents/embedspeed);
# TLSQ_EMBED_MODE=bitwise restores the bit-identical training features.
_blocks.EMBED_MODE["mode"] = os.environ.get("TLSQ_EMBED_MODE", "numex")
_blocks.EDGE_TABLE["on"] = os.environ.get("TLSQ_EDGE_TABLE", "1") == "1"  # array edge table (bitwise values; dicts on request)

lt = I.lt
EPS_TRAIN, T_TRAIN_UM = 11.45, 350.0
# Other substrate stacks (agents/v4stack, author decision 2026-11-03): the physics prior is solved at the requested (eps_r, thickness)
# and the frozen networks correct it; the geometric embedding and the global inputs stay at the training stack (Si 11.45, 350 um).
# Tested on the 15 held-out DoE designs: Si at 10-1000 um, and eps_r 3.8-24 at 350 um. Learned-prior entries (the Fast tier) cannot
# solve a prior: off the stack they keep the (1 + eps_r) / (1 + 11.45) rescaling of their training-stack prediction, with a warning.
TESTED_THICKNESS_UM = (10.0, 1000.0)
TESTED_EPS_R = (3.8, 24.0)
CALIBRATED_ROUTE = "calibrated: physics prior solved at the requested stack, networks and embedding at the training stack"
CALIBRATED_CAVEAT = (
    "calibrated substrate route: the physics prior is solved at your stack; the networks were trained on Si (eps_r 11.45), 350 um "
    "only and correct it unchanged. A bounded-error route, not training-stack accuracy: on 15 held-out DoE designs (networks trained on the TRAIN split) "
    "the mutual entries were 0.21-1.02 % (MAE 0.07-0.33 fF, RMSE 0.09-0.45 fF) on every tested stack, against 0.23 % (0.080 / "
    "0.111 fF) on the training stack; self terms 0.96-1.22 % (MAE 0.48-0.79, RMSE 0.52-0.86 fF) at 10-50 um Si and 0.89 % "
    "(0.92 / 1.07 fF) at 1000 um, against 0.24 % (0.18 / 0.22 fF)")
FAST_ROUTE = "fast: eps-rescaled training-stack prediction; thickness not modelled"
FAST_WARNING = ("Fast does not solve the prior at your stack: it rescales its 350 um silicon answer by (1 + eps_r) / (1 + 11.45) and "
                "ignores thickness. Measured off the stack: 6.4 % median error (MAE 1.7 fF, RMSE 2.4 fF), 13 % (3.4 / 4.7 fF) across Si "
                "thickness 10-1000 um; use the flagship (Accurate) model for other substrates (prior solved at your eps_r and thickness: 0.48 % "
                "(0.13 / 0.18 fF) on the same solves).")


def on_training_stack(eps_r: float, thickness_um: float) -> bool:
    return abs(float(eps_r) - EPS_TRAIN) <= 1e-9 and abs(float(thickness_um) - T_TRAIN_UM) <= 1e-9


def tested_stack(eps_r: float, thickness_um: float) -> bool:
    """The (eps_r, thickness) combinations the calibrated route was scored on: Si (11.45) at 10-1000 um, or eps_r 3.8-24 at 350 um."""
    e, t = float(eps_r), float(thickness_um)
    si = abs(e - EPS_TRAIN) <= 1e-9 and TESTED_THICKNESS_UM[0] <= t <= TESTED_THICKNESS_UM[1]
    return si or (abs(t - T_TRAIN_UM) <= 1e-9 and TESTED_EPS_R[0] <= e <= TESTED_EPS_R[1])


def substrate_route(eps_r: float, thickness_um: float, solves_prior: bool = True) -> tuple[dict, list]:
    """(substrate record, warnings) of a request. solves_prior=False: a learned-prior entry (the Fast tier), eps-rescaled off the stack."""
    e, t = float(eps_r), float(thickness_um)
    if not (np.isfinite(e) and e >= 1.0):
        raise ValueError(f"substrate eps_r must be a finite number >= 1 (got {eps_r!r})")
    if not (np.isfinite(t) and t > 0):
        raise ValueError(f"substrate thickness must be a finite number > 0 um (got {thickness_um!r})")
    train = {"eps_r": EPS_TRAIN, "thickness_um": T_TRAIN_UM}
    rec = {"eps_r": e, "thickness_um": t, "training_stack": train, "embedding_stack": dict(train), "tested": True}
    if on_training_stack(e, t):
        rec.update(route="training stack", prior_solved_at=dict(train) if solves_prior else None, eps_rescale_factor=1.0)
        return rec, []
    if not solves_prior:
        rec.update(route=FAST_ROUTE, prior_solved_at=None, eps_rescale_factor=(1.0 + e) / (1.0 + EPS_TRAIN), tested=False,
                   caveat="eps rescaling of the training-stack prediction, thickness not modelled; on 285 off-stack Q3D solves of 15 "
                   "held-out designs 6.36 % (MAE 1.74 fF, RMSE 2.39 fF): 13.1 % (3.43 / 4.67 fF) across Si thickness 10-1000 um, "
                   "1.44 % (0.51 / 0.73 fF) across eps_r 3.8-24 at 350 um, against 1.29 % (0.48 / 0.71 fF) on the training stack")
        return rec, [FAST_WARNING]
    rec.update(route=CALIBRATED_ROUTE, prior_solved_at={"eps_r": e, "thickness_um": t}, eps_rescale_factor=None,
               tested=tested_stack(e, t), caveat=CALIBRATED_CAVEAT)
    w = []
    if not rec["tested"]:
        w.append(f"substrate eps_r {e:g}, {t:g} um is outside the tested range of the calibrated route (Si eps_r 11.45 at "
                 f"{TESTED_THICKNESS_UM[0]:g}-{TESTED_THICKNESS_UM[1]:g} um, or eps_r {TESTED_EPS_R[0]:g}-{TESTED_EPS_R[1]:g} at "
                 f"{T_TRAIN_UM:g} um): untested, its error is unknown")
    return rec, w



def substrate_line(res: dict) -> str:
    """One Markdown line of a result's substrate route for the Space reports."""
    sub = res.get("substrate_route") or {}
    head = f"**Substrate** {res.get('substrate')} (eps_r {res['eps_r']:g}), {res['thickness_um']:g} µm: "
    r = sub.get("route", "training stack")
    if r == "training stack":
        return head + "the training stack"
    if r == FAST_ROUTE:
        return head + f"{r} (factor (1+eps_r)/(1+11.45) = {sub['eps_rescale_factor']:.4f}). {FAST_WARNING}"
    return head + r + ("" if sub.get("tested") else " (UNTESTED combination)") + f". {sub.get('caveat', '')}"

PRIOR_IMPLS = ("frozen", "cpu_fast")
_HMAT0 = dict(proxy.HMAT)
_STATE = {"prior_impl": "frozen", "patches": [], "mode": None, "stack": None}
_CACHE: dict = {}


def gnn_names(reg) -> list[str]:
    return [k for k, v in (reg.data.get("gnn_models") or {}).items() if v.get("status", "available") == "available"]


def set_prior_impl(impl: str) -> None:
    """frozen: the solver as frozen. cpu_fast: scripts/geomfast/fastmom.py (vectorised H-matrix kernel) with ACA eps 1e-6 and
    PCG tol 1e-8 (agents/geomfast/FASTPRIOR_STATUS.md); process-wide."""
    if impl not in PRIOR_IMPLS:
        raise ValueError(f"prior_impl {impl!r} not in {PRIOR_IMPLS}")
    if impl == _STATE["prior_impl"]:
        return
    from . import fastmom
    if impl == "cpu_fast":
        fastmom.install()
        proxy.HMAT.update(eps=1e-6, tol=1e-8)
    else:
        fastmom.uninstall()
        proxy.HMAT.clear()
        proxy.HMAT.update(_HMAT0)
    _STATE["prior_impl"] = impl


VALUE_PRESERVING = ("rejtail_fv",)  # bit-identical priors
VALUE_CHANGING = ("priorv2m_sf", "priorv2m", "priorv2")  # change the prior: need prior.value_changing_approved in the registry entry
# priorv2.installed() arguments of each value-changing patch: v2m_sf = the corrected prior with substrate far field (default since
# 2026-10-20); v2m = the corrected prior with the half-space far field (before 2026-10-20; reproduction); v2 = version 2 (no margin
# widening), the prior of the paper's "retrained, version 2" ablation (paper Space only)
V2_KWARGS = {"priorv2m_sf": dict(policy="fc", dip=True, margin=True), "priorv2m": dict(policy="fc", dip=True, margin=True),
             "priorv2": dict(policy="fc", dip=True, margin=False)}
# far-pair form on top of the v2 pieces: geomgnn.farfield = scripts/farfield/farfield.py verbatim, arm b of agents/farfield (slab dipole
# for signal far pairs in different ground pockets, half-space dipole inside a pocket); adopted by the user 2026-10-20
FAR_ARM = {"priorv2m_sf": "b"}
# TLSQ_FAR_FIELD=halfspace: serve priorv2m wherever priorv2m_sf is listed (the previous corrected prior, process-wide; reproduction)
FAR_FIELD_ENV = "TLSQ_FAR_FIELD"
PRIOR_PATCHES = VALUE_PRESERVING + VALUE_CHANGING


def resolve_patches(patches) -> list:
    """The registry patches with the TLSQ_FAR_FIELD override applied ('halfspace': priorv2m_sf -> priorv2m)."""
    v = os.environ.get(FAR_FIELD_ENV, "").strip().lower()
    if v not in ("", "substrate", "halfspace"):
        raise ValueError(f"{FAR_FIELD_ENV}={v!r}: use 'substrate' (default) or 'halfspace'")
    return sorted("priorv2m" if (p == "priorv2m_sf" and v == "halfspace") else p for p in (patches or []))


SPEED_MODES = ("exact", "exact_compiled", "certified")  # geomgnn.fastprior_pkg modes (agents/fastprior_pkg/RESULTS.md)


def speed_mode(pr: dict):
    """Solver mode of the prior: TLSQ_GNN_PRIOR_MODE (opt-in override, e.g. 'certified' or 'frozen') else registry prior.mode;
    None / 'frozen' = the vendor solver as frozen (no fastprior_pkg)."""
    m = os.environ.get("TLSQ_GNN_PRIOR_MODE") or pr.get("mode")
    if m in (None, "", "frozen"):
        return None
    if m not in SPEED_MODES:
        raise ValueError(f"prior mode {m!r} not in {SPEED_MODES} (or 'frozen')")
    return m


def _v2m_value_patch(F, P, FL, patch="priorv2m"):
    """fastprior_pkg value hook: prior v2m (geomgnn.priorv2 installed(policy='fc', dip=True, margin=True)) on top of the
    solver components (patch="priorv2": version 2, margin=False); returns the undo callable."""
    from . import priorv2
    assert priorv2.F is F and priorv2.P is P, "priorv2 and fastprior_pkg must patch the same vendor modules"
    return _install_v2(priorv2, F, P, patch)


def _install_v2(priorv2, F, P, patch):
    """priorv2.installed(**V2_KWARGS[patch]) and, for priorv2m_sf, the far-pair arm on top (geomgnn.farfield); returns the undo."""
    ctx = priorv2.installed(**V2_KWARGS[patch])
    ctx.__enter__()
    undo_far = None
    if patch in FAR_ARM:
        from . import farfield
        try:
            undo_far = farfield.install(FAR_ARM[patch], F, P, priorv2)
        except BaseException:
            ctx.__exit__(None, None, None)
            raise

    def undo():
        if undo_far is not None:
            undo_far()
        ctx.__exit__(None, None, None)
    return undo


def _teardown() -> None:
    from . import rejtail_fv
    if _STATE.get("mode"):
        from . import fastprior_pkg
        fastprior_pkg.uninstall()  # also undoes its value patch
    elif set(V2_KWARGS) & set(_STATE["patches"]):
        _STATE.pop("v2undo")()
    rejtail_fv.uninstall()
    _STATE.update(patches=[], mode=None, stack=None)


def set_prior_stack(pr: dict, threads: int = 2) -> None:
    """Process-wide prior stack of a registry prior: patches (rejtail_fv | priorv2m) and the solver mode (fastprior_pkg).
    rejtail_fv: geomgnn.rejtail_fv (value-preserving rejection-tail fix; fastprior_pkg's fv replaces it, so not with a mode).
    priorv2m:   prior v2m (agents/priorv2/FINALIZATION_V2.json): priorv2.installed(policy='fc', dip=True, margin=True), as
                fastprior_pkg's value_patch when a mode is set; it replaces _window_solve itself, so never with rejtail_fv.
    mode:       fastprior_pkg.install(mode, cpus=threads, target='geomgnn', value_patch=v2m) + warmup() (numba compile)."""
    patches = resolve_patches(pr.get("patches"))
    mode = speed_mode(pr)
    bad = set(patches) - set(PRIOR_PATCHES)
    if bad:
        raise ValueError(f"unknown prior patches {bad}; known: {PRIOR_PATCHES}")
    v2 = [p for p in patches if p in V2_KWARGS]
    if v2 and "rejtail_fv" in patches:
        raise ValueError("priorv2m already contains the rejection-tail handling; do not combine it with rejtail_fv")
    if len(v2) > 1:
        raise ValueError(f"one prior version per stack, got {v2}")
    if mode and "rejtail_fv" in patches:
        raise ValueError("fastprior_pkg contains fv; list rejtail_fv only without a prior mode")
    unapproved = [p for p in patches if p in VALUE_CHANGING and not pr.get("value_changing_approved")]
    if unapproved:
        raise ValueError(f"value-changing prior patches {unapproved} need prior.value_changing_approved in the registry")
    key = (tuple(patches), mode, int(threads))
    if key == _STATE.get("stack"):
        return
    _teardown()
    if mode:
        from . import fastprior_pkg as fp
        fp.install(mode=mode, cpus=int(threads), workers=os.environ.get("TLSQ_PRIOR_WORKERS"), target="geomgnn", blas_threads=int(threads),
                   value_patch=(lambda F, P, FL: _v2m_value_patch(F, P, FL, v2[0])) if v2 else None, warm=False)
        if mode in ("exact_compiled", "certified"):
            _STATE["warmup_s"] = fp.warmup()
    else:
        from . import priorv2, rejtail_fv
        if "rejtail_fv" in patches:
            rejtail_fv.install()
        if v2:
            _STATE["v2undo"] = _install_v2(priorv2, fastprior, proxy, v2[0])
    _STATE.update(patches=patches, mode=mode, stack=key, pr=dict(pr))
    if mode:
        from . import winpool
        winpool.install()  # macOS: the prior's windows on spawned workers (numerically exact; TLSQ_WINDOW_PROCS=0 turns it off)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def surrogate_file(reg, name: str, model_dir=None) -> Path | None:
    """The learned prior checkpoint of an entry (bundle copy named in info.json "surrogate", else the registry path)."""
    lp = reg["gnn_models"][name].get("learned_prior")
    if not lp:
        return None
    md = os.environ.get("TLSQ_MODEL_DIR") if model_dir is None else model_dir
    if md and (Path(md) / name / "info.json").exists():
        info = json.loads((Path(md) / name / "info.json").read_text())
        if info.get("surrogate"):
            return Path(md) / name / info["surrogate"]
    return reg.root / lp["checkpoint"]["path"]


def checkpoint_files(reg, name: str, model_dir=None) -> list[Path]:
    """Bundle copies (<model_dir>/<name>/info.json) or the registry's source paths; SHA256 checked against the registry."""
    m = reg["gnn_models"][name]
    md = os.environ.get("TLSQ_MODEL_DIR") if model_dir is None else model_dir
    if md and (Path(md) / name / "info.json").exists():
        info = json.loads((Path(md) / name / "info.json").read_text())
        files = [Path(md) / name / f for f in info["files"]]
    else:
        files = [reg.root / c["path"] for c in m["checkpoints"]]
    return files


def available(name: str, reg, model_dir=None) -> bool:
    s = surrogate_file(reg, name, model_dir)
    return all(f.exists() for f in checkpoint_files(reg, name, model_dir)) and (s is None or s.exists())


def load(name: str, reg, model_dir=None, threads: int | None = None) -> I.Predictor:
    m = reg["gnn_models"][name]
    threads = int(threads or os.environ.get("TLSQ_NUM_THREADS") or 2)
    key = (str(reg.path), name, str(os.environ.get("TLSQ_MODEL_DIR") if model_dir is None else model_dir), threads)
    if key in _CACHE:
        _apply_prior(m, threads)
        return _CACHE[key]
    files = checkpoint_files(reg, name, model_dir)
    for f, c in zip(files, m["checkpoints"], strict=True):
        if _sha(f) != c["sha256"]:
            raise RuntimeError(f"{name}: {f.name} SHA256 differs from the registry (frozen checkpoint {c['sha256'][:12]})")
    P = I.Predictor(files, threads=threads)
    if P.arm != m["arm"]:
        raise RuntimeError(f"{name}: checkpoints are arm {P.arm!r}, registry says {m['arm']!r}")
    pr = m.get("prior")
    lpr = m.get("learned_prior")
    phys = P.models[0][0].get("schema", {}).get("physics", {})
    P.surrogate = None
    if P.direct:
        if pr or lpr:
            raise RuntimeError(f"{name}: direct arm has no prior, registry lists one")
    elif lpr:
        if pr or P.fast or [float(x) for x in lpr["ppg"]] != [1.5, 3.0]:
            raise RuntimeError(f"{name}: a learned prior replaces the 1.5 / 3 ppg field solves of a full_gnn arm only ({P.arm}, {lpr})")
        sf = surrogate_file(reg, name, model_dir)
        if _sha(sf) != lpr["checkpoint"]["sha256"]:
            raise RuntimeError(f"{name}: {sf.name} SHA256 differs from the registry (learned prior {lpr['checkpoint']['sha256'][:12]})")
        from . import learned_prior
        model, prep, _ = learned_prior.load(sf)
        prep.floor_ff = lpr.get("floor_ff")
        P.surrogate = (model, prep)
    else:
        if not pr or abs(float(pr["alpha"]) - float(phys.get("alpha"))) > 0 or (pr["method"] == "local_gap") != (phys.get("prior_method") == "local_gap") \
                or [float(x) for x in pr["ppg"]] != ([0.75, 1.5] if P.fast else [1.5, 3.0]):
            raise RuntimeError(f"{name}: registry prior {pr} does not match the checkpoint physics {phys}")
    _apply_prior(m, threads)
    _CACHE[key] = P
    return P


def _apply_prior(m: dict, threads: int = 2) -> None:
    """Process-wide prior settings of a gnn_models entry (on every load: entries may differ, the patches are global)."""
    pr = m.get("prior")
    if pr:
        set_prior_impl(pr.get("prior_impl", "frozen"))
        set_prior_stack(pr, threads)


STARTUP: dict = {}  # preload record (reported by /models)
_TLS = threading.local()


def _thread_limits(threads: int) -> None:
    """numba and OpenMP thread counts are per thread: apply fastprior_pkg's limits once in every thread that predicts (the
    preload runs in the main thread, requests in the server's worker threads)."""
    if not _STATE.get("mode") or getattr(_TLS, "key", None) == (_STATE["stack"], threads):
        return
    from . import fastprior_pkg
    fastprior_pkg._set_threads(int(threads))
    _TLS.key = (_STATE["stack"], threads)


def preload(name: str, reg, example=None) -> dict:
    """Server start: load the default model (SHA check, prior stack, numba warmup) and, if given, predict one small example so
    the first user request runs at normal speed. Synchronous (about 20-30 s on the 2-vCPU Space), before the queue opens. The
    learned-prior entries (no prior stack) are loaded and run on the example too."""
    t0 = time.perf_counter()
    load(name, reg)
    STARTUP.update(model=name, load_and_warmup_s=time.perf_counter() - t0, numba_warmup_s=_STATE.get("warmup_s"), mode=_STATE.get("mode"))
    also = [k for k in gnn_names(reg) if k != name and reg["gnn_models"][k].get("learned_prior") and available(k, reg)]
    if example is not None and Path(example).exists():
        from .layout import prepare
        t = time.perf_counter()
        predict_gnn(prepare(str(example)), name, reg)
        STARTUP["example_s"] = time.perf_counter() - t
        for k in also:
            t = time.perf_counter()
            predict_gnn(prepare(str(example)), k, reg)
            STARTUP.setdefault("also_s", {})[k] = time.perf_counter() - t
    else:
        for k in also:
            load(k, reg)
    STARTUP["total_s"] = time.perf_counter() - t0
    print("preload " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in STARTUP.items()}), flush=True)
    return dict(STARTUP)


def scene_from_prepared(lay) -> Scene:
    """The Scene scene_from_gds builds (ground = net 0, signal bodies in file order, chip = ground bounding box, training stack),
    from the prepared ground and conductor bodies."""
    G = lay.ground
    chip = shapely.box(*G.bounds)
    conds = [Conductor("G", "ground", G)] + [Conductor(n, "signal", g) for n, g in zip(lay.names, lay.conductors)]
    return Scene(conds, chip, chip.difference(G), dict(DEFAULT_STACK), {})


def predict_scene(P: I.Predictor, sc: Scene, alpha=None, local_gap=None, stack=None) -> dict:
    """infer.Predictor.from_gds after the scene (same statements). stack: (eps_r, thickness_um) the prior is solved at (default the
    scene's, i.e. the training stack); the embedding always uses the scene's stack."""
    eps_r, t_um = stack or (sc.stack["substrate_eps_r"], sc.stack["substrate_thickness_um"])
    physics = P.models[0][0].get("schema", {}).get("physics", {})
    if alpha is None:
        alpha = physics.get("alpha")
    if local_gap is None:
        local_gap = physics.get("prior_method") == "local_gap"
    proxy.NUM_THREADS = P.threads
    if set(V2_KWARGS) & set(_STATE["patches"]):
        from . import priorv2
        priorv2.clear()  # its per-window traces would otherwise grow for the life of the server
    start = time.perf_counter()
    t1 = time.perf_counter()
    from . import overlap as OV
    fut = OV.start(sc, P.threads) if (getattr(P, "surrogate", None) is None and not P.direct and local_gap and OV.enabled(sc, P.threads)) else None
    row = None if fut else lt.add_tokens(sc, scene_graph(sc))  # overlap: the embedding runs in the worker during the priors
    t2 = time.perf_counter()
    priors = {}
    if getattr(P, "surrogate", None) is not None:
        from . import learned_prior
        priors = learned_prior.priors(*P.surrogate, row)
    elif not P.direct:
        old = fastprior.ALPHA_GAP_SHARE
        try:
            if alpha is not None:
                fastprior.ALPHA_GAP_SHARE = alpha
            from . import levelpar
            lvls = [.75, 1.5] if P.fast else [1.5, 3.]
            if local_gap and levelpar.enabled(sc, lvls, P.threads):  # the levels concurrently (spawned workers, same stack and statements)
                priors.update(levelpar.solve(sc, lvls, P.threads, alpha, (eps_r, t_um))[0])
                lvls = []
            for p in lvls:
                if local_gap:
                    from ._tlsq.embed.fastprior_loc import micro_mom_loc
                    with proxy.substrate("slab_open", t_um), _quiet_mixed_cast():
                        mat, _, _ = micro_mom_loc(sc.conductors[0].geom, [c.geom for c in sc.conductors[1:]], eps_r=eps_r, ppg=p)
                    priors[p] = -mat
                else:
                    if stack is not None and not on_training_stack(eps_r, t_um):
                        raise ValueError("the off-stack prior needs the local-gap prior method")
                    priors[p] = -I.CapacitanceMoM(max_cells=10**9, mesh="lite", cells_per_gap=p,
                                                  substrate="slab_open").compute(sc, list(row["edges"]))["matrix"]
        finally:
            fastprior.ALPHA_GAP_SHARE = old
    t3 = time.perf_counter()
    extra = {}
    if fut is not None:
        row, embed_s = fut.result()
        extra = dict(overlap=True, geometry_worker=embed_s, geometry_wait_after_prior=time.perf_counter() - t3)
        t3 = time.perf_counter()
    C = P.from_features(row, priors)
    t4 = time.perf_counter()
    return dict(C=C, names=row["names"], timing=dict(scene=t1 - start, geometry=t2 - t1, prior=t3 - t2, network_and_pack=t4 - t3,
                                                     total=t4 - start, **extra))


@contextlib.contextmanager
def _quiet_mixed_cast():
    """fastprior_pkg's mixed Cholesky casts the system to float32 first; on some windows the cast overflows and numpy warns
    'invalid value / overflow encountered in cast' before the package falls back to the float64 solve (expected; the result is checked
    by iterative refinement). Silence exactly that warning, from that module, during the prior call only."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="(invalid value|overflow) encountered in cast", category=RuntimeWarning,
                                module=r"squadds\.ml\.capacitance\._core\.fastprior_pkg")
        yield


def physical_checks(C: np.ndarray) -> dict:
    n = len(C)
    off = C[~np.eye(n, dtype=bool)]
    sig = C[1:, 1:]
    return {"symmetric_max_abs": float(np.max(np.abs(C - C.T))), "offdiag_all_negative": bool(np.all(off < 0)),
            "signal_block_min_eig": float(np.linalg.eigvalsh(sig).min()) if n > 1 else None}


def physical_banner(res: dict) -> list[str]:
    """Report lines that put a failed physical check first."""
    w = next((w for w in res.get("warnings") or [] if w.startswith("physical check failed")), None)
    return [f"> **Not a physical matrix.** {w}", ""] if w else []


def physical_warning(chk: dict, C: np.ndarray) -> str | None:
    """Readable warning when the checks of physical_checks fail (None when the matrix passes)."""
    bad = []
    if not chk["symmetric_max_abs"] <= 1e-6 * max(1.0, float(np.abs(C).max())):
        bad.append(f"not symmetric (max |C - C^T| {chk['symmetric_max_abs']:.3g} fF)")
    if not chk["offdiag_all_negative"]:
        bad.append(f"{int(np.sum(C[~np.eye(len(C), dtype=bool)] >= 0))} off-diagonal entries are not negative")
    if chk["signal_block_min_eig"] is not None and not chk["signal_block_min_eig"] > 0:
        bad.append(f"the signal block is not positive definite (smallest eigenvalue {chk['signal_block_min_eig']:.4g} fF)")
    if not bad:
        return None
    return ("physical check failed: " + "; ".join(bad) + ". This matrix is not physical: do not use it. A model that solves the physics "
            "(flagship / Accurate) is the safer choice, especially outside the trained component families.")


def predict_gnn(prepared, name: str, reg, substrate="Si (cryogenic, training)", thickness_um: float = 350.0, model_dir=None,
                threads: int | None = None) -> dict:
    """Full N x N Maxwell matrix (fF; ground row/column 0, self terms learned) of a prepared layout with a gnn_models entry."""
    m = reg["gnn_models"][name]
    eps, t_um = reg.eps_of(substrate), float(thickness_um)
    sub, sub_w = substrate_route(eps, t_um, solves_prior=not m.get("learned_prior"))
    stack = None if on_training_stack(eps, t_um) else (eps, t_um)
    t = time.perf_counter()
    P = load(name, reg, model_dir, threads)
    _thread_limits(P.threads)
    load_s = time.perf_counter() - t
    sc = scene_from_prepared(prepared)
    try:
        r = predict_scene(P, sc, stack=stack if sub["eps_rescale_factor"] is None else None)
    except AssertionError as e:
        raise ValueError(f"{name}: the prior or the features of this layout are not finite (no matrix returned): {e}") from e
    C = np.asarray(r["C"], float)
    if stack is not None and sub["eps_rescale_factor"] is not None:
        C = C * sub["eps_rescale_factor"]
    chk = physical_checks(C)
    labels = ["G"] + list(prepared.names)
    pr = m.get("prior")
    lpr = m.get("learned_prior")
    warnings = [w for w in prepared.warnings if "beyond the validated range" not in w]
    if m.get("warning"):
        warnings.insert(0, m["warning"])
    pw = physical_warning(chk, C)
    if pw:
        warnings.insert(0, pw)
    warnings += sub_w
    tm = r["timing"]
    return {"names": labels, "matrix_fF": C.tolist(), "units": "fF",
            "convention": "Maxwell matrix: diagonal = self (learned), off-diagonal = -mutual; row/column 0 = ground",
            "self_capacitance": "predicted by the network (learned positive row-sum closure; trained). The ground self term is "
                                "trained at low weight and depends on the chip boundary: indicative only",
            "model": name, "model_title": m.get("title"), "recipe": "GNN", "arm": m["arm"], "role": m.get("role"),
            "ensemble": m.get("ensemble"), "weights_sha256": [c["sha256"] for c in m["checkpoints"]],
            "prior": ({**pr, "prior_impl": _STATE["prior_impl"], "patches": _STATE.get("patches", []), "mode": _STATE.get("mode") or "frozen",
                       "numba_warmup_s": _STATE.get("warmup_s")} if pr else None),
            "learned_prior": ({"name": lpr["name"], "sha256": lpr["checkpoint"]["sha256"], "ppg": lpr["ppg"]} if lpr else None),
            "package_version": f"geomgnn {__version__}",
            "registry_version": reg["version"],
            "substrate": substrate if not isinstance(substrate, (int, float)) else f"custom eps={eps:g}", "eps_r": eps, "thickness_um": t_um,
            "substrate_route": sub, "training_stack": {"eps_r": EPS_TRAIN, "thickness_um": T_TRAIN_UM},
            "timing_s": {"load_s": load_s, "geometry_s": tm["geometry"], "prior_s": tm["prior"], "network_s": tm["network_and_pack"],
                         "total_s": tm["total"], "threads": P.threads},
            "accuracy": m.get("accuracy") or {}, "checks": chk, "trust": None, "expected_error": {}, "warnings": warnings,
            "stored": False}
