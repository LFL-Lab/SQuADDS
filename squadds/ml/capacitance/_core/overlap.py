"""Embedding concurrently with the prior solves (agents/embedspeed). Both read only the geometry, so the embedding row
(scene_graph + localtok.add_tokens, the same functions) is computed in one spawned worker process while the caller runs the
prior solves, and joined before the network: bitwise the same row and priors as the sequential path.

The worker is spawned (never forked: macOS Accelerate; the prior forks its own window workers on Linux) once per process
and reused. TLSQ_OVERLAP=0 turns it off; layouts with fewer than TLSQ_OVERLAP_MIN_N conductors (default 30), and machines with fewer than
TLSQ_OVERLAP_MIN_THREADS threads (default 4), stay sequential.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context

_EX: dict = {}


def enabled(sc, threads: int = 2) -> bool:
    """On for >= TLSQ_OVERLAP_MIN_N conductors (default 30) and >= TLSQ_OVERLAP_MIN_THREADS threads (default 4): measured on 2 x86
    CPUs the two stages compete for the cores (3 x 3 lattice 141 -> 153 s with the overlap), from 16 they gain (grid_full 609 -> 572 s)."""
    return (os.environ.get("TLSQ_OVERLAP", "1") != "0" and len(sc.conductors) >= int(os.environ.get("TLSQ_OVERLAP_MIN_N", "30"))
            and int(threads) >= int(os.environ.get("TLSQ_OVERLAP_MIN_THREADS", "4")))


def _settings() -> dict:
    from squadds.ml.capacitance._core._tlsq.uge import blocks

    return {"EMBED_MODE": dict(blocks.EMBED_MODE), "EMBED_CACHE": dict(blocks.EMBED_CACHE),
            "FAR_DECIMATION": dict(blocks.FAR_DECIMATION)}


def _apply(threads: int, st: dict):
    from squadds.ml.capacitance._core._tlsq.embed import proxy
    from squadds.ml.capacitance._core._tlsq.uge import blocks

    proxy.NUM_THREADS = threads
    for k, v in st.items():
        getattr(blocks, k).update(v)


def _init(threads: int, paths: list):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "TLSQ_NUM_THREADS"):
        os.environ[k] = str(threads)
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)


def _embed(sc, threads: int, st: dict) -> tuple:
    import time

    from squadds.ml.capacitance._core._tlsq.uge import localtok as lt
    from squadds.ml.capacitance._core._tlsq.uge.graph import scene_graph

    _apply(threads, st)
    t = time.perf_counter()
    return lt.add_tokens(sc, scene_graph(sc)), time.perf_counter() - t


def _executor(threads: int) -> ProcessPoolExecutor:
    if _EX.get("key") != threads:
        if _EX.get("ex") is not None:
            _EX["ex"].shutdown(wait=False, cancel_futures=True)
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))  # the directory holding squadds
        main = sys.modules["__main__"]
        saved = main.__dict__.pop("__file__", None)  # spawn would otherwise re-run the calling script in the worker
        try:
            from ._spawn import boot
            ini, args = boot(_init, threads, [here])
            ex = ProcessPoolExecutor(1, mp_context=get_context("spawn"), initializer=ini, initargs=args)
            ex.submit(int, 0).result()  # start the worker now, while __file__ is hidden
        finally:
            if saved is not None:
                main.__file__ = saved
        _EX.update(key=threads, ex=ex)
    return _EX["ex"]


def start(sc, threads: int) -> Future:
    """Future of (embedding row, worker seconds) of scene sc, computed in the worker process."""
    return _executor(threads).submit(_embed, sc, threads, _settings())
