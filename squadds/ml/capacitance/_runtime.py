"""Layout preparation, layout hashing and the process-wide thread settings shared by the predictor and the priors."""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import subprocess
import tempfile
import threading
from pathlib import Path

TRAIN_EPS_R, TRAIN_THICKNESS_UM = 11.45, 350.0
GROUND_LAYER, CONDUCTOR_LAYER = (1, 0), (1, 10)
LOCK = threading.RLock()  # the prior stack and thread pools of the frozen stage are process-wide
_USER_ENV = {k: os.environ.get(k) for k in ("TLSQ_WINDOW_PROCS", "TLSQ_LEVEL_PROCS")}
_TP: list = []


def prepare_layout(
    layout, ground_layers=(GROUND_LAYER,), conductor_layers=(CONDUCTOR_LAYER,), max_conductors: int = 1000, **kwargs
):
    """A ``PreparedLayout`` (ground net + conductor bodies, names C1..Cn in file order) from a GDS path, a gdstk Library or
    Cell, or a ``PreparedLayout`` (returned unchanged). Extra keyword arguments go to ``_core.layout.prepare`` (crop margins,
    selections, ignored layers, cell, ``ground_pieces``).

    Ground and conductors on one layer (the same ``ground_layers`` and ``conductor_layers``, or a single-layer file): the metal
    is split into connected pieces and the ground is detected automatically, with a warning; check the numbered figure
    (``inspect_layout``) and, if it is wrong, pass ``ground_pieces=[piece numbers]``."""
    from ._core.layout import PreparedLayout, prepare

    if isinstance(layout, PreparedLayout):
        return layout
    if isinstance(layout, (str, os.PathLike)):
        return prepare(
            str(layout),
            ground_layers=ground_layers,
            conductor_layers=conductor_layers,
            max_conductors=max_conductors,
            **kwargs,
        )
    import gdstk

    if isinstance(layout, gdstk.Cell):
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        lib.add(layout, *layout.dependencies(True))
        layout = lib
    if isinstance(layout, gdstk.Library):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "layout.gds"
            layout.write_gds(str(p))
            lay = prepare(
                str(p),
                ground_layers=ground_layers,
                conductor_layers=conductor_layers,
                max_conductors=max_conductors,
                **kwargs,
            )
        lay.source = "<gdstk.Library>"
        return lay
    raise TypeError(
        f"cannot read a layout from {type(layout).__name__}: pass a GDS path, a gdstk Library / Cell or a PreparedLayout"
    )


def inspect_layout(
    layout,
    ground_layers=(GROUND_LAYER,),
    conductor_layers=(CONDUCTOR_LAYER,),
    ground_pieces=None,
    figsize=(6.4, 6.4),
    **kwargs,
) -> dict:
    """The roles a prediction would use, before predicting: ``pieces`` (id, role, name, area, bounding box; empty when ground
    and conductors are on separate layers), ``ground_pieces``, ``reason``, ``confidence`` ("high" / "low" / "user"),
    ``warnings``, the ``PreparedLayout`` and a labelled matplotlib ``figure`` (ground grey, conductors coloured, every piece
    numbered). If the ground is wrong, predict with ``ground_pieces=[...]`` using the numbers of the figure."""
    lay = prepare_layout(layout, ground_layers, conductor_layers, ground_pieces=ground_pieces, **kwargs)
    gd = lay.info.get("ground_detection") or {}
    return {
        "detected": bool(gd),
        "pieces": gd.get("pieces", []),
        "ground_pieces": gd.get("ground_pieces"),
        "reason": gd.get("reason"),
        "confidence": gd.get("confidence"),
        "names": lay.names,
        "warnings": list(lay.warnings),
        "prepared": lay,
        "figure": lay.render(figsize=figsize),
    }


def layout_hash(lay) -> str:
    """sha256 of the prepared geometry (WKB of ground and conductors) and the conductor names."""
    import shapely

    h = hashlib.sha256(repr(list(lay.names)).encode())
    for g in [lay.ground, *lay.conductors]:
        h.update(shapely.to_wkb(g))
    return h.hexdigest()


MAX_DEFAULT_THREADS = 32
GB_PER_THREAD = 2.5  # macOS window workers are processes: a 67-conductor grid peaks ~1.4 GB higher per extra thread


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects containers, SLURM and taskset
    except AttributeError:
        return os.cpu_count() or 1


def _total_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:  # noqa: BLE001
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (AttributeError, ValueError, OSError):
            return None


def default_threads() -> int:
    """Threads used when none are given: SQUADDS_CAPML_THREADS if set, else every CPU this process may use, at most one per
    2.5 GB of RAM and at most 32."""
    env = os.environ.get("SQUADDS_CAPML_THREADS", "").strip()
    if env:
        return max(1, int(env))
    n = min(_usable_cpus(), MAX_DEFAULT_THREADS)
    ram = _total_ram_gb()
    if ram:
        n = min(n, int(ram // GB_PER_THREAD))
    return max(1, n)


def resolve_threads(threads: int | None) -> int:
    return default_threads() if threads is None else max(1, int(threads))


def configure(threads: int | None) -> int:
    """One knob for every pool of the pipeline: torch, BLAS (threadpoolctl), and on macOS the spawned window workers of the
    prior (TLSQ_WINDOW_PROCS = threads) and the level workers (TLSQ_LEVEL_PROCS: only from 4 threads). None means
    ``default_threads()``. Explicit environment settings of the user win."""
    import torch

    threads = resolve_threads(threads)
    torch.set_num_threads(threads)
    try:
        from threadpoolctl import threadpool_limits

        _TP[:] = [threadpool_limits(threads)]
    except Exception:  # noqa: BLE001  (Accelerate ignores it)
        pass
    os.environ["TLSQ_NUM_THREADS"] = str(threads)
    if _USER_ENV["TLSQ_WINDOW_PROCS"] is None and platform.system() == "Darwin":  # elsewhere: the Space default (off)
        os.environ["TLSQ_WINDOW_PROCS"] = str(threads if threads >= 2 else 0)
    if _USER_ENV["TLSQ_LEVEL_PROCS"] is None:
        os.environ["TLSQ_LEVEL_PROCS"] = "auto" if threads >= 4 else "0"
    return threads


@functools.lru_cache(maxsize=1)
def machine() -> str:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
                ).stdout.strip()
                or cpu
            )
        except Exception:  # noqa: BLE001
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return f"{cpu}, {os.cpu_count()} CPUs, {platform.system()} {platform.machine()}"
