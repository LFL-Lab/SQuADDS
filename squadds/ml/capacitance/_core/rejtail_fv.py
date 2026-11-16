"""rejtail fix fv (agents/rejtail/RESULTS.md, scripts/rejtail/rt.py; user approval 2026-10-17): value-preserving removal of
the rejection tail of the windowed micro-MoM prior (`fastprior._window_solve`, all_rhs=False, layouts with > 8 conductors).

A window attempt whose mesh leaves a metal without a panel is always rejected, whatever its solve returns. The frozen code
still solves it, and above DENSE_MAX panels the H-matrix PCG cannot converge: the panel-less metal has a zero right-hand
side, so the residual is 0/0 = NaN, and it runs 1,000 iterations before raising. fv meshes such attempts but does not
solve them. If the loop ends without an accepted attempt, the result the unpatched code returns is rebuilt by solving
the deferred dense attempts lazily:
  keep_last_solved=True  (frozen geomfast stage): the latest attempt whose solve succeeded, newest first;
  keep_last_solved=False (tlsq_infer vendor):     the last attempt's own solve, else the same RuntimeError.
Every accepted solve is the same call on the same mesh, so the result is bit-identical (621 layouts in RESULTS.md;
deploy/tests/test_gnn_space.py re-checks it). all_rhs=True calls (single window, N <= 8) run the original function.
The same file is shipped as geomgnn/rejtail_fv.py and tlsq_infer/_vendor/rejtail_fv.py; only the import line differs.
"""
from __future__ import annotations

import time

import numpy as np

from ._tlsq.embed import fastprior as F
from ._tlsq.embed import proxy as P

KEEP_LAST_SOLVED = True
_ORIG = {}


def _solve(c, wh, o, metals, cols, eps_r):
    """One attempt solved exactly as the original loop does (windowed: LDL^T dense, H-matrix PCG above DENSE_MAX).
    Returns (R, assemble seconds)."""
    t = time.perf_counter()
    t_asm = 0.0
    if len(c) <= P.DENSE_MAX:
        A = P._assemble(c, wh, P.rect_self(wh[:, 0], wh[:, 1]), eps_r, P.rect_potential)
        t_asm = time.perf_counter() - t
        Bm = (o[:, None] == cols[None, :]).astype(float)
        Q = F._sym_solve(A, Bm)
    else:
        Q = P.mom_solve_fast(c, wh, o, len(metals), eps_r)[:, cols]
    return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1), t_asm


def window_solve(metals: list, n_core: int, h: float, budget: float, eps_r: float, all_rhs: bool, levels: tuple = ()) -> tuple:
    if all_rhs:
        return _ORIG["window_solve"](metals, n_core, h, budget, eps_r, all_rhs, levels)
    cols = np.arange(1, 1 + n_core)
    t_mesh = t_asm = t_sol = 0.0
    hh, cap, refined, majority = h, budget, 0, False
    hist = []  # per attempt, oldest first: ("R", R) | ("err",) | ("mesh", c, wh, o) deferred dense | ("doomed",) deferred H-matrix
    while True:
        t = time.perf_counter()
        if not refined:
            c, wh, o, hh, n_raw = F._lite_mesh(metals, hh, cap, n_core, levels)
        elif levels:
            c, wh, o, hh, n_raw = F._graded_mesh(metals, n_core, hh, levels, majority)
        else:
            c, wh, o, hh, n_raw = P.mesh_flagship(metals, hh, cap, majority=majority)
        t_mesh += time.perf_counter() - t
        if not np.all(np.bincount(o, minlength=len(metals)) > 0):
            hist.append(("doomed",) if len(c) > P.DENSE_MAX else ("mesh", c, wh, o))
            ok = False
        else:
            t = time.perf_counter()
            try:
                R, ta = _solve(c, wh, o, metals, cols, eps_r)
                t_asm += ta
                hist.append(("R", R))
                ok = True
            except np.linalg.LinAlgError:
                hist.append(("err",))
                ok = False
            t_sol += time.perf_counter() - t
        if ok or majority:
            break
        if refined < F.LITE_REFINE:
            hh, cap, refined = hh / 2, np.inf, refined + 1
        else:
            majority = True
    if ok:
        R = hist[-1][1]
    else:
        R = None
        t = time.perf_counter()
        for e in reversed(hist if KEEP_LAST_SOLVED else hist[-1:]):
            if e[0] == "R":
                R = e[1]
                break
            if e[0] == "mesh":
                try:
                    R = _solve(e[1], e[2], e[3], metals, cols, eps_r)[0]
                    break
                except np.linalg.LinAlgError:
                    continue
        t_sol += time.perf_counter() - t
        if R is None:
            raise RuntimeError(f"window solve failed: no positive-definite system after {refined} refinements (h={hh:.4g} um); "
                               "the window has no valid solution at this mesh")
    return R, {"h_um": hh, "panels": len(c), "raw": n_raw, "refined": refined, "majority": majority, "mesh_s": t_mesh,
               "assemble_s": t_asm, "solve_s": t_sol}


def install() -> None:
    """Replace fastprior._window_solve (a module global, looked up at call time, also in the forked window workers)."""
    if not _ORIG:
        _ORIG["window_solve"] = F._window_solve
        F._window_solve = window_solve


def uninstall() -> None:
    if _ORIG:
        F._window_solve = _ORIG.pop("window_solve")


def installed() -> bool:
    return bool(_ORIG)
