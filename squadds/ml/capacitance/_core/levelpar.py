"""Mesh levels of the prior concurrently (agents/embedspeed). The two thin-film solves (e.g. 1.5 and 3 panels per gap) are
independent; on macOS each runs its windows serially and keeps only ~2-5 cores busy, so the extra levels run in spawned
worker processes that install the same prior stack (geomgnn.api.set_prior_stack with the recorded registry prior and
threads) and execute the same statements as the in-process loop; the caller solves the first level meanwhile.

TLSQ_LEVEL_PROCS: auto (default: TLSQ_LEVEL_MIN_N <= N <= TLSQ_LEVEL_MAX_N, default 30 and 150, on macOS or with >= 8 threads),
1 (always), 0 (never).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_EX: dict = {}


def enabled(sc, levels, threads: int = 2) -> bool:
    from . import api as A
    v = os.environ.get("TLSQ_LEVEL_PROCS", "auto")
    if v == "0" or len(levels) < 2 or not A._STATE.get("pr"):
        return False
    if v == "1":
        return True
    n = len(sc.conductors)  # measured (Mac, 18 threads): a gain at N = 67 (18.1 -> 11.9 s with spawned windows), a loss from
    # N = 277 on (69 -> 75 s; grid_full 194 -> 238 s) where the spawned windows already fill the cores
    # Linux x86, 16 CPUs (forked windows): 5q 15.1 -> 12.0 s, 9q 76.0 -> 46.6 s, 1q 7.8 -> 10.3 s (spawn cost); never below 8 threads
    mid = int(os.environ.get("TLSQ_LEVEL_MIN_N", "30")) <= n <= int(os.environ.get("TLSQ_LEVEL_MAX_N", "150"))
    return mid and (sys.platform == "darwin" or int(threads) >= int(os.environ.get("TLSQ_LEVEL_MIN_THREADS", "8")))


def _init(paths, impl, pr, threads):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "TLSQ_NUM_THREADS"):
        os.environ[k] = str(threads)
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(threads)
    from squadds.ml.capacitance._core import api as A
    A.set_prior_impl(impl)
    A.set_prior_stack(pr, threads)
    A._thread_limits(threads)


def solve_one(sc, p, threads, alpha, stack=None):
    """The statements of the prior loop (geomgnn.api.predict_scene / paper.solve_levels) for one level; stack = (eps_r, thickness_um)
    the prior is solved at (default the scene's)."""
    from squadds.ml.capacitance._core import api as A
    from squadds.ml.capacitance._core._tlsq.embed import fastprior, proxy
    from squadds.ml.capacitance._core._tlsq.embed.fastprior_loc import micro_mom_loc
    proxy.NUM_THREADS = threads
    if set(A.V2_KWARGS) & set(A._STATE["patches"]):
        from squadds.ml.capacitance._core import priorv2
        priorv2.clear()
    old = fastprior.ALPHA_GAP_SHARE
    t = time.perf_counter()
    try:
        if alpha is not None:
            fastprior.ALPHA_GAP_SHARE = alpha
        eps_r, t_um = stack or (sc.stack["substrate_eps_r"], sc.stack["substrate_thickness_um"])
        with proxy.substrate("slab_open", t_um), A._quiet_mixed_cast():
            mat, _, _ = micro_mom_loc(sc.conductors[0].geom, [c.geom for c in sc.conductors[1:]], eps_r=eps_r, ppg=p)
    finally:
        fastprior.ALPHA_GAP_SHARE = old
    return -mat, time.perf_counter() - t


def _executor(n, threads):
    from . import api as A
    key = (n, threads, A._STATE["prior_impl"], repr(A._STATE["pr"]))
    if _EX.get("key") != key:
        if _EX.get("ex") is not None:
            _EX["ex"].shutdown(wait=False, cancel_futures=True)
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))  # the directory holding squadds
        main = sys.modules["__main__"]
        saved = main.__dict__.pop("__file__", None)  # spawn would otherwise re-run the calling script in the worker
        try:
            from ._spawn import boot
            ini, args = boot(_init, [here], A._STATE["prior_impl"], A._STATE["pr"], threads)
            ex = ProcessPoolExecutor(n, mp_context=get_context("spawn"), initializer=ini, initargs=args)
            list(ex.map(int, range(n)))
        finally:
            if saved is not None:
                main.__file__ = saved
        _EX.update(key=key, ex=ex)
    return _EX["ex"]


def solve(sc, levels, threads, alpha, stack=None) -> tuple[dict, dict]:
    """{ppg: P} and {ppg: seconds} with levels[1:] in the workers and levels[0] here."""
    ex = _executor(len(levels) - 1, threads)
    futs = [ex.submit(solve_one, sc, p, threads, alpha, stack) for p in levels[1:]]
    first = solve_one(sc, levels[0], threads, alpha, stack)
    res = [first] + [f.result() for f in futs]
    return {p: r[0] for p, r in zip(levels, res)}, {f"{p:g}": r[1] for p, r in zip(levels, res)}
