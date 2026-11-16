"""Window solves of the prior on spawned workers (agents/embedspeed; macOS). The vendor's window pool forks, which macOS
cannot do with Accelerate, so on macOS fastprior_pkg runs the windows serially (a few cores busy). Here the vendor's own
micro_mom runs with `workers` > 1 and its ProcessPoolExecutor / get_context("fork") are swapped, for that call only, for a
spawn pool whose workers install the same prior stack (geomgnn.api.set_prior_stack with the recorded registry prior), copy the
running call's layout (fastprior._LITE, the k-d tree rebuilt) and the module settings, and run fastprior_pkg's window job.
Each window is the same statements; BLAS threads per window differ (one per worker), so the result is numerically exact,
not bitwise (gate: <= 1e-9 relative vs the serial windows).

TLSQ_WINDOW_PROCS: auto (default: macOS, CPUs from TLSQ_NUM_THREADS / os.cpu_count()), <n> (force n workers), 0 (off).
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_ON: dict = {"n": 0}


def workers() -> int:
    v = os.environ.get("TLSQ_WINDOW_PROCS", "auto")
    if v == "0":
        return 0
    if v != "auto":
        return int(v)
    return (os.cpu_count() or 1) if sys.platform == "darwin" else 0


def _init(paths, impl, pr, state):
    os.environ["TLSQ_WINDOW_PROCS"] = "0"
    os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                                         "TLSQ_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)
    import shapely
    import torch
    torch.set_num_threads(1)
    from squadds.ml.capacitance._core import api as A
    from squadds.ml.capacitance._core._tlsq.embed import fastprior as F
    from squadds.ml.capacitance._core._tlsq.embed import proxy
    A.set_prior_impl(impl)
    A.set_prior_stack(pr, 1)
    A._thread_limits(1)
    proxy.NUM_THREADS = 1
    proxy.SUBSTRATE.clear()
    proxy.SUBSTRATE.update(state["substrate"])
    for k in ("ALPHA_GAP_SHARE", "LITE_WINDOW_BUDGET", "LITE_REFINE", "LITE_GROWTH", "FORMULA_STEP_UM", "ROW_SUM_EPS"):
        setattr(F, k, state["F"][k])
    F.LITE_WINDOW_PPG[:] = state["F"]["LITE_WINDOW_PPG"]
    F.LITE_MAX_CORE[:] = state["F"]["LITE_MAX_CORE"]
    lite = dict(state["lite"])
    lite["tree"] = shapely.STRtree(lite["conds"])
    F._LITE.clear()
    F._LITE.update(lite)


class _SpawnPool(ProcessPoolExecutor):
    def __init__(self, max_workers, mp_context=None):
        from . import api as A
        from ._tlsq.embed import fastprior as F
        from ._tlsq.embed import proxy
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))  # the directory holding squadds
        state = dict(substrate=dict(proxy.SUBSTRATE), lite={k: v for k, v in F._LITE.items() if k != "tree"},
                     F={k: getattr(F, k) for k in ("ALPHA_GAP_SHARE", "LITE_WINDOW_BUDGET", "LITE_REFINE", "LITE_GROWTH",
                                                    "FORMULA_STEP_UM", "ROW_SUM_EPS")}
                     | {"LITE_WINDOW_PPG": list(F.LITE_WINDOW_PPG), "LITE_MAX_CORE": list(F.LITE_MAX_CORE)})
        main = sys.modules["__main__"]
        self._saved = main.__dict__.pop("__file__", None)  # spawn would otherwise re-run the calling script
        from ._spawn import boot
        ini, args = boot(_init, [here], A._STATE["prior_impl"], A._STATE["pr"], state)
        super().__init__(max_workers, mp_context=get_context("spawn"), initializer=ini, initargs=args)

    def shutdown(self, *a, **k):
        try:
            super().shutdown(*a, **k)
        finally:
            if self._saved is not None:
                sys.modules["__main__"].__file__ = self._saved
                self._saved = None


class _Serial:
    """In-process stand-in for the other pools of the call (formula_prior's rows): what macOS runs today."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False

    def map(self, fn, *its, **k):
        return map(fn, *its)


def _pool(max_workers, mp_context=None):
    """Spawned workers for the windows of a call with >= TLSQ_WINDOW_MIN_CORES windows (default 5; measured on the Mac: the
    1q lattice's 2 windows 3.4 -> 5.9 s with the spawn, the 9q lattice's 9 windows 23.8 -> 18.1 s); in-process otherwise."""
    from ._tlsq.embed import fastprior as F
    big = len(F._LITE.get("cores", ())) >= int(os.environ.get("TLSQ_WINDOW_MIN_CORES", "5"))
    return _SpawnPool(max_workers) if F._LITE and big else _Serial()


def install() -> bool:
    """Wrap the installed stack's micro_mom so that its windows run on spawned workers (idempotent; macOS by default)."""
    from . import api as A
    from . import fastprior_pkg as fp
    from ._tlsq.embed import fastprior as F
    n = workers()
    if n < 2 or not A._STATE.get("pr") or not A._STATE.get("mode") or "micro_mom" not in fp._ORIG:
        return False
    if getattr(F.micro_mom, "_winpool", False):
        _ON["n"] = n
        return True
    vendor = fp._ORIG["micro_mom"]
    sig = fp._ORIG["micro_mom_sig"]

    def micro_mom(*args, **kw):
        ba = sig.bind(*args, **kw)
        if _ON["n"] < 2:
            return fp._micro_mom(*args, **kw)
        ba.arguments["workers"] = _ON["n"]
        fp._PARENT[0] = os.getpid()
        old = F.ProcessPoolExecutor, F.get_context
        F.ProcessPoolExecutor, F.get_context = _pool, (lambda method=None: None)
        try:
            return vendor(*ba.args, **ba.kwargs)
        finally:
            F.ProcessPoolExecutor, F.get_context = old

    micro_mom._winpool = True
    F.micro_mom = micro_mom
    _ON["n"] = n
    return True
