"""Physics priors: thin-film method-of-moments capacitance matrices of a layout (no learned parameter).

- ``physics_prior``: one solve at p = 0.375 panels per gap (the paper's cheap-physics comparator).
- ``optimized_physics_prior``: C^R = R(1.5, 3) = exp(2 log C(3) - log C(1.5)), entry by entry, the physics prior of the paper
  model and of QCap.

Both run on the "corrected" prior stack (prior v2m + substrate far field, compiled exact solver; gap share alpha fitted once on
TRAIN), exactly as the models compute them, on a substrate of ``eps_r`` and ``thickness_um`` (default the training stack, Si 11.45,
350 µm; the thin-film kernel is a dielectric slab of that thickness with vacuum below). Priors computed on macOS are not bit-identical to the released ones (HPC x86; up
to ~2.4e-3 relative on some layouts): never mix them into released data.

TODO (other priors of the paper, not served yet):
- uncorrected prior (v1, frozen solver + value-preserving rejection-tail fix; stack ``uncorrected``)
- version 2 (no window-margin widening; stack ``version2``)
- corrected prior with the half-space far field (v2m before 2026-10-20; stack ``corrected_halfspace``)
- single solves at 0.75 / 1.5 / 3 panels per gap
- Richardson pair R(0.75, 1.5) (the paper's balanced variant)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import _runtime as RT

STACK = "corrected"


@dataclass
class PriorResult:
    names: list[str]
    C_fF: np.ndarray
    solves_fF: dict = field(default_factory=dict)
    levels_ppg: list[float] = field(default_factory=list)
    kind: str = ""
    prior_version: str = ""
    stack: dict = field(default_factory=dict)
    alpha: float = 0.0
    timing_s: dict = field(default_factory=dict)
    threads: int = 2
    machine: str = ""
    layout_hash: str = ""
    units: str = "fF"

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.C_fF, index=self.names, columns=self.names)

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "C_fF": self.C_fF.tolist(),
            "solves_fF": {f"{k:g}": v.tolist() for k, v in self.solves_fF.items()},
            "levels_ppg": self.levels_ppg,
            "kind": self.kind,
            "prior_version": self.prior_version,
            "stack": self.stack,
            "alpha": self.alpha,
            "timing_s": self.timing_s,
            "threads": self.threads,
            "machine": self.machine,
            "layout_hash": self.layout_hash,
            "units": self.units,
        }


def _registry():
    from .models import registry_view

    return registry_view({}, ".", "squadds-priors")


def solve(
    layout,
    levels,
    threads: int | None = None,
    eps_r: float = RT.TRAIN_EPS_R,
    thickness_um: float = RT.TRAIN_THICKNESS_UM,
    **prepare_kwargs,
) -> tuple:
    """(prepared layout, {ppg: positive-mutual matrix P = -C}, {ppg: seconds}, stack record, hash) under the corrected stack, solved
    on a substrate of eps_r and thickness_um."""
    from ._core import api as A
    from ._core import paper as PA
    from .models import MANIFEST
    from .predictor import _PRI, _get, _put, quiet

    sub, w = A.substrate_route(eps_r, thickness_um)
    stack = None if A.on_training_stack(eps_r, thickness_um) else (sub["eps_r"], sub["thickness_um"])
    lay = RT.prepare_layout(layout, **prepare_kwargs)
    lhash = RT.layout_hash(lay)
    reg = _registry()
    with RT.LOCK, quiet():
        threads = RT.configure(threads)
        PA.apply_stack(reg, {"stack": STACK}, threads)
        key = (lhash, tuple(float(p) for p in levels), repr(A._STATE.get("stack")), threads, stack)
        P, tl = _get(_PRI, key), {}
        if P is None:
            sc = A.scene_from_prepared(lay)
            P, tl = PA.solve_levels(sc, [float(p) for p in levels], threads, stack)
            _put(_PRI, key, P)
        st = MANIFEST["prior_stacks"][STACK]
        rec = {
            "name": STACK,
            "version": st["version"],
            "patches": list(A._STATE.get("patches", [])),
            "mode": A._STATE.get("mode"),
            "substrate": {"eps_r": sub["eps_r"], "thickness_um": sub["thickness_um"], "tested": sub["tested"]},
        }
    for m in w:
        warnings.warn(
            m.replace("of the calibrated route", "of the calibrated substrate study"), UserWarning, stacklevel=4
        )
    return lay, P, tl, rec, lhash


def _result(kind, layout, levels, combine, threads, eps_r, thickness_um, prepare_kwargs) -> PriorResult:
    from ._core import paper as PA
    from ._core import x1_core as X

    t = time.perf_counter()
    threads = RT.resolve_threads(threads)
    lay, P, tl, rec, lhash = solve(layout, levels, threads, eps_r, thickness_um, **prepare_kwargs)
    C = np.asarray(combine(P), float)
    return PriorResult(
        names=["G"] + list(lay.names),
        C_fF=C,
        solves_fF={p: X.to_maxwell(P[p]) for p in levels},
        levels_ppg=list(levels),
        kind=kind,
        prior_version=rec["version"],
        stack=rec,
        alpha=PA.ALPHA,
        timing_s={"total_s": time.perf_counter() - t, "levels_s": tl, "cached": not tl},
        threads=int(threads),
        machine=RT.machine(),
        layout_hash=lhash,
    )


def physics_prior(
    layout,
    threads: int | None = None,
    eps_r: float = RT.TRAIN_EPS_R,
    thickness_um: float = RT.TRAIN_THICKNESS_UM,
    **prepare_kwargs,
) -> PriorResult:
    """One thin-film solve at 0.375 panels per gap (corrected stack) on a substrate of eps_r and thickness_um: Maxwell matrix in
    fF, ground first."""
    from ._core import x1_core as X

    return _result(
        "physics_prior p=0.375",
        layout,
        [0.375],
        lambda P: X.to_maxwell(P[0.375]),
        threads,
        eps_r,
        thickness_um,
        prepare_kwargs,
    )


def optimized_physics_prior(
    layout,
    threads: int | None = None,
    eps_r: float = RT.TRAIN_EPS_R,
    thickness_um: float = RT.TRAIN_THICKNESS_UM,
    **prepare_kwargs,
) -> PriorResult:
    """C^R = R(1.5, 3) of the corrected stack (two solves, Richardson-extrapolated in log, every entry) on a substrate of eps_r
    and thickness_um: Maxwell matrix in fF, ground first; both solves are returned in ``solves_fF``."""
    from ._core import x1_core as X

    return _result(
        "optimized_physics_prior R(1.5,3)",
        layout,
        [1.5, 3.0],
        lambda P: X.c0(P, 1.5, 3.0),
        threads,
        eps_r,
        thickness_um,
        prepare_kwargs,
    )
