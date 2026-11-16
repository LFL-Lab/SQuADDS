"""GDS -> full N x N Maxwell capacitance matrix with a released model.

The steps are those of the tl-squadds Spaces (``geomgnn.paper.predict`` / ``geomgnn.flagship.predict``), statement for
statement: scene from the prepared layout -> geometric embedding (the model's far-sample setting) -> thin-film priors under the
model's prior stack (or the learned prior surrogate) -> networks. The embedding and the priors are cached in memory by the
layout's geometry hash, so a repeated layout skips them.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from . import _runtime as RT
from .models import CapacitanceModel, load_model, no_tier
from .result import CapacitanceResult, physical_checks, physical_warning

_EMB: OrderedDict = OrderedDict()
_PRI: OrderedDict = OrderedDict()
_CACHE_SIZE = [int(os.environ.get("SQUADDS_CAPML_CACHE", "32"))]
_CACHE_BYTES = [int(os.environ.get("SQUADDS_CAPML_CACHE_BYTES", str(2**30)))]  # per cache
_NBYTES: dict = {}  # id(cache) -> {key: estimated bytes}


@contextlib.contextmanager
def quiet():
    """Silence numba's performance hints of the compiled solver (no effect on values)."""
    with warnings.catch_warnings():
        try:
            from numba.core.errors import NumbaPerformanceWarning

            warnings.simplefilter("ignore", NumbaPerformanceWarning)
        except ImportError:
            pass
        yield


@atexit.register
def _teardown() -> None:
    """Undo the process-wide prior stack before interpreter shutdown (its patches hold module references)."""
    import sys

    A = sys.modules.get("squadds.ml.capacitance._core.api")
    if A is not None and A._STATE.get("stack") is not None:
        with contextlib.suppress(Exception):
            A._teardown()


def _warn(msg: str) -> None:
    warnings.warn(msg, UserWarning, stacklevel=4)


def clear_cache() -> None:
    _EMB.clear()
    _PRI.clear()
    _NBYTES.clear()


def _nbytes(v) -> int:
    """Estimated memory of a cached value: arrays / tensors by nbytes, containers and Python objects by sys.getsizeof."""
    import sys

    nb = getattr(v, "nbytes", None)
    if isinstance(nb, int):
        return nb
    if isinstance(v, dict):
        return sys.getsizeof(v) + sum(_nbytes(k) + _nbytes(x) for k, x in v.items())
    if isinstance(v, (list, tuple, set)):
        return sys.getsizeof(v) + sum(_nbytes(x) for x in v)
    return sys.getsizeof(v)


def _get(c: OrderedDict, k):
    if k in c:
        c.move_to_end(k)
        return c[k]
    return None


def _put(c: OrderedDict, k, v) -> None:
    if _CACHE_SIZE[0] <= 0:
        return
    nb = _NBYTES.setdefault(id(c), {})
    c[k] = v
    c.move_to_end(k)
    nb[k] = _nbytes(v)
    while c and (len(c) > _CACHE_SIZE[0] or sum(nb.values()) > _CACHE_BYTES[0]):
        nb.pop(c.popitem(last=False)[0], None)


def _stack_record(a: dict) -> tuple[dict | None, str | None]:
    from ._core import api as A
    from ._core import paper as PA
    from .models import MANIFEST

    if a.get("stack") is None:
        return None, None
    st = MANIFEST["prior_stacks"][a["stack"]]
    return (
        {
            "name": a["stack"],
            "version": st.get("version"),
            "patches": A._STATE.get("patches", []),
            "mode": A._STATE.get("mode") or "frozen",
            "alpha": PA.ALPHA,
            "method": "local_gap micro-MoM (slab)",
        },
        st.get("version"),
    )


def run_scene(reg, name: str, sc, lhash: str, threads: int, stack: tuple | None = None) -> dict:
    """geomgnn.paper.predict_scene with the embedding row and the priors taken from / put into the in-memory caches. stack:
    (eps_r, thickness_um) the prior is solved at (None: the training stack); the embedding always uses the training stack."""
    from ._core import overlap as OV
    from ._core import paper as PA
    from ._core._tlsq.uge import blocks as B
    from ._core._tlsq.uge.graph import scene_graph

    a = PA.arms(reg)[name]
    obj = PA.load(name, reg, "", threads)
    PA.apply_stack(reg, a, threads)
    emb = {} if os.environ.get("TLSQ_EMBEDDING_EXACT") == "1" else (a.get("embedding") or {})
    old = dict(B.FAR_DECIMATION)
    B.FAR_DECIMATION.update(k=int(emb.get("far_k", 0)), d_far=float(emb.get("d_far_um", 100.0)))
    try:
        from ._core import api as A

        need = [] if (a["kind"] == "gnn" and (obj.direct or obj.surrogate is not None)) else PA.levels_of(a)
        ekey = (
            lhash,
            B.FAR_DECIMATION["k"],
            B.FAR_DECIMATION["d_far"],
            B.EMBED_MODE["mode"],
            B.EDGE_TABLE["on"],
            threads,
        )
        pkey = (lhash, tuple(need), repr(A._STATE.get("stack")), threads, stack)
        row = _get(_EMB, ekey)
        P = _get(_PRI, pkey) if need else {}
        hits = dict(embedding=row is not None, prior=bool(need) and P is not None)
        t0 = time.perf_counter()
        fut = OV.start(sc, threads) if (row is None and P is None and need and OV.enabled(sc, threads)) else None
        if row is None and fut is None:
            row = PA.lt.add_tokens(sc, scene_graph(sc))
        t1 = time.perf_counter()
        tl = {}
        if P is None:
            P, tl = PA.solve_levels(sc, need, threads, stack)
        t2 = time.perf_counter()
        extra = {}
        if fut is not None:
            row, embed_s = fut.result()
            extra = dict(overlap=True, geometry_worker=embed_s, geometry_wait_after_prior=time.perf_counter() - t2)
            t2 = time.perf_counter()
        _put(_EMB, ekey, row)
        if need:
            _put(_PRI, pkey, P)
        C = np.asarray(PA.combine(a, obj, row, P), float)
        t3 = time.perf_counter()
    finally:
        B.FAR_DECIMATION.clear()
        B.FAR_DECIMATION.update(old)
    return dict(
        C=C,
        names=row["names"],
        row=row,
        priors=P,
        cache_hits=hits,
        timing=dict(geometry=t1 - t0, prior=t2 - t1, prior_levels=tl, network=t3 - t2, total=t3 - t0, **extra),
    )


def _released_names(text: str) -> str:
    """The frozen ``_core`` warnings with the released model names."""
    return text.replace("Fast does not solve", "QCap-Fast does not solve").replace(
        "the flagship (Accurate) model", "QCap"
    )


def _substrate(eps_r, thickness_um, learned_prior: bool) -> tuple[tuple | None, dict, list]:
    """(stack the prior is solved at or None for the training stack, substrate record, warnings). Tiers that solve a prior solve it
    at (eps_r, thickness_um) (the calibrated route); QCap-Fast's learned prior only knows the training stack, so off the stack
    its training-stack prediction is rescaled by (1 + eps_r) / (1 + 11.45) and a warning is raised."""
    from ._core import api as A

    sub, w = A.substrate_route(eps_r, thickness_um, solves_prior=not learned_prior)
    w = [_released_names(x) for x in w]
    return (None if A.on_training_stack(eps_r, thickness_um) else (sub["eps_r"], sub["thickness_um"])), sub, w


class CapacitancePredictor:
    """``CapacitancePredictor(load_model("QCap")).predict("chip.gds")`` -> CapacitanceResult.

    threads: CPU threads per prediction (default: every core this process may use, at most one per 2.5 GB of RAM and 32;
        ``SQUADDS_CAPML_THREADS`` sets it too). Results are deterministic for a given thread count and machine, so pass the same
        ``threads`` to reproduce a result exactly.
    embed_cache_dir: optional on-disk cache of per-body embedding blocks (bit-identical on the same machine).
    """

    def __init__(
        self,
        model: CapacitanceModel | str | None = None,
        threads: int | None = None,
        embed_cache_dir=None,
        tier=None,
        **load_kwargs,
    ):
        no_tier(tier)
        if model is None or isinstance(model, str):
            model = load_model(model or "QCap", **load_kwargs)
        self.model = model
        self._auto_threads = threads is None
        self.threads = RT.resolve_threads(threads)
        self.reg = model.registry()
        if embed_cache_dir is not None:
            from ._core._tlsq.uge import blocks as B

            B.EMBED_CACHE["dir"] = str(embed_cache_dir)

    def __repr__(self) -> str:
        return f"CapacitancePredictor({self.model.name!r}, threads={self.threads})"

    def predict(
        self,
        layout,
        threads: int | None = None,
        eps_r: float = RT.TRAIN_EPS_R,
        thickness_um: float = RT.TRAIN_THICKNESS_UM,
        workers: int | None = None,
        **prepare_kwargs,
    ):
        """layout: GDS path, gdstk Library / Cell, PreparedLayout, or a list of these (a list returns a list; ``workers`` > 1
        predicts it on that many processes, each with ``threads`` threads; by default the cores are split between them). Extra keyword arguments go to the layout
        preparation (ground_layers, conductor_layers, margin_gaps, select, ...).

        eps_r, thickness_um: the substrate. Off the training stack (Si, 11.45, 350 µm) QCap
        solves the physics prior at this stack and correct it with the unchanged networks (the calibrated route, recorded in
        ``result.substrate``); QCap-Fast rescales its training-stack prediction by (1 + eps_r) / (1 + 11.45) and warns.
        Combinations outside the tested range (Si at 10-1000 µm, or eps_r 3.8-24 at 350 µm) warn."""
        if threads is None:
            threads = self.threads
            if self._auto_threads and workers and int(workers) > 1 and isinstance(layout, (list, tuple)):
                threads = max(1, threads // int(workers))  # split the cores between the worker processes
        threads = int(threads)
        _substrate(eps_r, thickness_um, self._learned_prior())
        if isinstance(layout, (list, tuple)):
            return self._batch(list(layout), threads, eps_r, thickness_um, workers, prepare_kwargs)
        return self._one(layout, threads, eps_r, thickness_um, prepare_kwargs)

    def _learned_prior(self) -> bool:
        from ._core import paper as PA

        return bool(PA.arms(self.reg)[self.model.name].get("learned_prior"))

    def _one(self, layout, threads, eps_r, thickness_um, prepare_kwargs) -> CapacitanceResult:
        from ._core import api as A
        from ._core import paper as PA

        at, sub, sub_w = _substrate(eps_r, thickness_um, self._learned_prior())
        t = time.perf_counter()
        lay = RT.prepare_layout(layout, **prepare_kwargs)
        prep_s = time.perf_counter() - t
        lhash = RT.layout_hash(lay)
        name = self.model.name
        a = PA.arms(self.reg)[name]
        with RT.LOCK, quiet():
            RT.configure(threads)
            t = time.perf_counter()
            PA.load(name, self.reg, "", threads)
            load_s = time.perf_counter() - t
            sc = A.scene_from_prepared(lay)
            try:
                r = run_scene(self.reg, name, sc, lhash, threads, None if sub["eps_rescale_factor"] is not None else at)
            except AssertionError as e:
                raise ValueError(
                    f"{self.model.name}: the prior or the features of this layout are not finite (no matrix returned): {e}"
                ) from e
            stack, version = _stack_record(a)
        C = r["C"] if at is None or sub["eps_rescale_factor"] is None else r["C"] * sub["eps_rescale_factor"]
        names = ["G"] + list(lay.names)
        if list(r["names"]) != names:
            raise RuntimeError(f"scene names {r['names']} differ from the prepared layout {names}")
        warnings = [w for w in lay.warnings if "beyond the validated range" not in w]
        if a.get("warning"):
            warnings.insert(0, a["warning"])
        chk = physical_checks(C)
        pw = physical_warning(chk)
        if pw:
            warnings.insert(0, pw)
            _warn(pw)
        warnings += sub_w
        gd = lay.info.get("ground_detection")
        for w in sub_w + [
            w
            for w in lay.warnings
            if gd and gd["method"] == "automatic" and w.startswith("ground and conductors share")
        ]:
            _warn(w)
        tm = r["timing"]
        timing = {
            "prepare_s": prep_s,
            "load_s": load_s,
            "geometry_s": tm["geometry"],
            "prior_s": tm["prior"],
            "prior_levels_s": tm["prior_levels"],
            "network_s": tm["network"],
            "total_s": tm["total"],
            "cache_hits": r["cache_hits"],
            "threads": threads,
        }
        if tm.get("overlap"):
            timing["overlap"] = True
        from . import _core

        return CapacitanceResult(
            names=names,
            matrix_fF=C,
            model=self.model.name,
            title=self.model.title,
            prior_version=version or ("learned prior surrogate" if a.get("learned_prior") else None),
            prior_stack=stack,
            weights_sha256=self.model.weights_sha256,
            learned_prior_sha256=self.model.learned_prior_sha256,
            timing_s=timing,
            threads=threads,
            machine=RT.machine(),
            warnings=warnings,
            substrate=sub,
            ground_detection=gd,
            checks=chk,
            source=str(lay.source),
            layout_hash=lhash,
            cached=r["cache_hits"]["embedding"] and (r["cache_hits"]["prior"] or not tm["prior_levels"]),
            package=f"squadds.ml.capacitance (_core = geomgnn {_core.__version__})",
        )

    def _batch(self, items, threads, eps_r, thickness_um, workers, prepare_kwargs) -> list[CapacitanceResult]:
        workers = min(int(workers or 1), len(items)) if items else 1
        if workers <= 1:
            return [self._one(x, threads, eps_r, thickness_um, prepare_kwargs) for x in items]
        lays = [RT.prepare_layout(x, **prepare_kwargs) for x in items]
        m = self.model
        from ._core._spawn import boot

        ini, args = boot(_worker_init, m, threads)
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=ini, initargs=args) as ex:
            return list(ex.map(_worker_predict, lays, [(eps_r, thickness_um)] * len(lays)))


_WORKER: dict = {}


def _worker_init(model, threads):
    RT._USER_ENV.update(TLSQ_WINDOW_PROCS="0", TLSQ_LEVEL_PROCS="0")  # one process per layout: no nested worker pools
    os.environ.update(RT._USER_ENV)
    _WORKER["p"] = CapacitancePredictor(model, threads=threads)


def _worker_predict(lay, sub):
    return _WORKER["p"].predict(lay, eps_r=sub[0], thickness_um=sub[1])
