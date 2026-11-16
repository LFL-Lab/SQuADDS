"""Exact nearest-sample distances (agents/embedspeed, CONSULT_HPC 3a): a 2-D k-d tree in numba whose squared distance is
the platform's cKDTree arithmetic (probed at first use against scipy: fma(dy, dy, dx * dx) on the arm64 wheels, plain
dx * dx + dy * dy elsewhere), queried in boundary order with a warm-start bound from the previous sample (triangle
inequality, inflated), so the minimum over samples, and its sqrt, are the same bits as cKDTree.query. If no variant
reproduces cKDTree on the probe, `available()` is False and callers keep cKDTree.
"""
from __future__ import annotations

import os
import threading

import numba as nb
import numpy as np
from llvmlite import ir
from numba import types
from numba.extending import intrinsic

LEAF = 32
WARM = 4  # boundary-order neighbours of the previous query's nearest sample evaluated first (an exact upper bound)
_STATE: dict = {}


@intrinsic
def _fma(typingctx, a, b, c):
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        d = ir.DoubleType()
        fn = builder.module.declare_intrinsic("llvm.fma", [d], ir.FunctionType(d, [d, d, d]))
        return builder.call(fn, args)

    return sig, codegen


@nb.njit(cache=True)
def _build(P, leaf):
    """Balanced k-d tree over P (m, 2): permutation, node boxes and ranges; node k has children 2k+1, 2k+2."""
    m = P.shape[0]
    nn = 1
    while (m + leaf - 1) // leaf > (nn + 1) // 2:
        nn = 2 * nn + 1
    perm = np.arange(m)
    lo = np.zeros(nn, np.int64)
    hi = np.zeros(nn, np.int64)
    box = np.empty((nn, 4))
    lo[0], hi[0] = 0, m
    for k in range(nn):
        a, b = lo[k], hi[k]
        if b > a:
            x0 = x1 = P[perm[a], 0]
            y0 = y1 = P[perm[a], 1]
            for t in range(a + 1, b):
                x, y = P[perm[t], 0], P[perm[t], 1]
                x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            box[k, 0], box[k, 1], box[k, 2], box[k, 3] = x0, y0, x1, y1
        else:
            box[k, 0] = box[k, 1] = np.inf
            box[k, 2] = box[k, 3] = -np.inf
        c = 2 * k + 1
        if c < nn:
            mid = (a + b) // 2
            if b - a > 1:
                d = 0 if box[k, 2] - box[k, 0] >= box[k, 3] - box[k, 1] else 1
                sub = perm[a:b].copy()
                order = np.argsort(P[sub, d], kind="mergesort")
                perm[a:b] = sub[order]
            lo[c], hi[c] = a, mid
            lo[c + 1], hi[c + 1] = mid, b
    return perm, lo, hi, box


@nb.njit(cache=True, inline="always")
def _mind2(box, k, x, y):
    dx = max(box[k, 0] - x, 0.0, x - box[k, 2])
    dy = max(box[k, 1] - y, 0.0, y - box[k, 3])
    return dx * dx + dy * dy


@nb.njit(nogil=True, cache=True, inline="always")
def _sqd(x, y, P, j, use_fma):
    dx, dy = x - P[j, 0], y - P[j, 1]
    return _fma(dy, dy, dx * dx) if use_fma else dx * dx + dy * dy


@nb.njit(nogil=True, cache=True)
def _kd(x, y, best, jb, P, perm, lo, hi, box, use_fma, stack):
    """Exact k-d search below the bound best (a squared distance already attained by sample jb, or inf)."""
    nn = lo.shape[0]
    stack[0] = 0
    sp = 1
    while sp:
        sp -= 1
        k = stack[sp]
        if lo[k] >= hi[k] or _mind2(box, k, x, y) * (1.0 - 1e-12) > best:
            continue
        c = 2 * k + 1
        if c >= nn:
            for t in range(lo[k], hi[k]):
                j = perm[t]
                s = _sqd(x, y, P, j, use_fma)
                if s < best:
                    best, jb = s, j
            continue
        d1, d2 = _mind2(box, c, x, y), _mind2(box, c + 1, x, y)
        if d1 <= d2:
            stack[sp] = c + 1
            stack[sp + 1] = c
        else:
            stack[sp] = c
            stack[sp + 1] = c + 1
        sp += 2
    return best, jb


@nb.njit(nogil=True, cache=True)
def _scan(x, y, best, jb, P, et, ej, a, b, t0, t1, use_fma):
    """min over the edge-list entries a..b with t in [t0, t1] (t sorted)."""
    lo_, hi_ = a, b
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if et[mid] < t0:
            lo_ = mid + 1
        else:
            hi_ = mid
    for i in range(lo_, b):
        if et[i] > t1:
            break
        s = _sqd(x, y, P, ej[i], use_fma)
        if s < best:
            best, jb = s, ej[i]
    return best, jb


@nb.njit(nogil=True, cache=True)
def _nearest_t(x, y, P, et, ej, a, b, tf, use_fma):
    """An attained upper bound: the better of the two list entries around t = tf."""
    lo_, hi_ = a, b
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if et[mid] < tf:
            lo_ = mid + 1
        else:
            hi_ = mid
    best, jb = np.inf, -1
    for i in (lo_ - 1, lo_):
        if a <= i < b:
            s = _sqd(x, y, P, ej[i], use_fma)
            if s < best:
                best, jb = s, ej[i]
    return best, jb


@nb.njit(nogil=True, cache=True)
def _query(X, P, perm, lo, hi, box, use_fma, warm, H, hv, hx, hy, ux, uy, L, nx, ny, eoff, et, ej, eps, delta):
    """Distances from every row of X to the nearest sample of P. Far field (H > 0): x outside the hull of P projects to a
    hull edge or vertex f at distance dH; every sample s satisfies |x - s|^2 >= dH^2 + |f - s|^2 (obtuse angle at f), so once
    a sample o attains s_o, only samples within rho = sqrt(s_o - dH^2 + slack) of f can be nearer; those lie in the edge's
    list (samples within delta of it). Otherwise, and whenever rho > delta, the k-d search (warm-started) decides."""
    m = X.shape[0]
    mp = P.shape[0]
    out = np.empty(m)
    stack = np.empty(128, np.int64)
    jp = -1
    e = 0
    for q in range(m):
        x, y = X[q, 0], X[q, 1]
        best = np.inf
        jb = -1
        done = False
        if H >= 3:
            reg = -1
            vtx = -1
            for _ in range(H + 2):
                t = (x - hx[e]) * ux[e] + (y - hy[e]) * uy[e]
                if t < 0.0:
                    ep = e - 1 if e > 0 else H - 1
                    if (x - hx[ep]) * ux[ep] + (y - hy[ep]) * uy[ep] > L[ep]:
                        vtx = e
                        break
                    e = ep
                    continue
                if t > L[e]:
                    en = e + 1 if e + 1 < H else 0
                    if (x - hx[en]) * ux[en] + (y - hy[en]) * uy[en] < 0.0:
                        vtx = en
                        break
                    e = en
                    continue
                if (x - hx[e]) * nx[e] + (y - hy[e]) * ny[e] > 0.0:
                    reg = e
                break
            if vtx >= 0:
                fx, fy = hx[vtx], hy[vtx]
                dh2 = (x - fx) * (x - fx) + (y - fy) * (y - fy)
                best = _sqd(x, y, P, hv[vtx], use_fma)
                jb = hv[vtx]
                rho2 = best - dh2 + 1e-9 * best + 1e-4 + 4.0 * eps * (np.sqrt(best) + 1.0)
                if rho2 <= delta * delta:
                    r = np.sqrt(max(rho2, 0.0))
                    ep = vtx - 1 if vtx > 0 else H - 1
                    best, jb = _scan(x, y, best, jb, P, et, ej, eoff[vtx], eoff[vtx + 1], -r, r, use_fma)
                    best, jb = _scan(x, y, best, jb, P, et, ej, eoff[ep], eoff[ep + 1], L[ep] - r, L[ep] + r, use_fma)
                    done = True
            elif reg >= 0:
                tf = (x - hx[reg]) * ux[reg] + (y - hy[reg]) * uy[reg]
                pp = (x - hx[reg]) * nx[reg] + (y - hy[reg]) * ny[reg]
                dh2 = pp * pp
                best, jb = _nearest_t(x, y, P, et, ej, eoff[reg], eoff[reg + 1], tf, use_fma)
                if jb >= 0:
                    rho2 = best - dh2 + 1e-9 * best + 1e-4 + 4.0 * eps * (np.sqrt(best) + 1.0)
                    if rho2 <= delta * delta:
                        r = np.sqrt(max(rho2, 0.0))
                        best, jb = _scan(x, y, best, jb, P, et, ej, eoff[reg], eoff[reg + 1], tf - r, tf + r, use_fma)
                        done = True
        if not done:
            if jp >= 0:
                for j in range(max(0, jp - warm), min(mp, jp + warm + 1)):
                    s = _sqd(x, y, P, j, use_fma)
                    if s < best:
                        best, jb = s, j
            best, jb = _kd(x, y, best, jb, P, perm, lo, hi, box, use_fma, stack)
        out[q] = np.sqrt(best)
        jp = jb
    return out


@nb.njit(cache=True)
def _hull_lists(P, hv, delta):
    H = hv.shape[0]
    hx, hy = np.empty(H), np.empty(H)
    ux, uy, L, nx, ny = np.empty(H), np.empty(H), np.empty(H), np.empty(H), np.empty(H)
    for i in range(H):
        hx[i], hy[i] = P[hv[i], 0], P[hv[i], 1]
    for i in range(H):
        k = i + 1 if i + 1 < H else 0
        dx, dy = hx[k] - hx[i], hy[k] - hy[i]
        L[i] = np.sqrt(dx * dx + dy * dy)
        ux[i], uy[i] = dx / L[i], dy / L[i]
        nx[i], ny[i] = uy[i], -ux[i]  # outward for a counter-clockwise hull
    m = P.shape[0]
    eps = 0.0
    cnt = np.zeros(H + 1, np.int64)
    db = delta + 0.01
    for i in range(H):
        for j in range(m):
            ax, ay = P[j, 0] - hx[i], P[j, 1] - hy[i]
            pp = ax * nx[i] + ay * ny[i]
            if pp > eps:
                eps = pp
            t = ax * ux[i] + ay * uy[i]
            if -db <= t <= L[i] + db and abs(pp) <= db:
                cnt[i + 1] += 1
    eoff = np.cumsum(cnt)
    et, ej = np.empty(eoff[H]), np.empty(eoff[H], np.int64)
    for i in range(H):
        k = eoff[i]
        for j in range(m):
            ax, ay = P[j, 0] - hx[i], P[j, 1] - hy[i]
            pp = ax * nx[i] + ay * ny[i]
            t = ax * ux[i] + ay * uy[i]
            if -db <= t <= L[i] + db and abs(pp) <= db:
                et[k], ej[k] = t, j
                k += 1
        o = np.argsort(et[eoff[i]:eoff[i + 1]], kind="mergesort")
        et[eoff[i]:eoff[i + 1]] = et[eoff[i]:eoff[i + 1]][o]
        ej[eoff[i]:eoff[i + 1]] = ej[eoff[i]:eoff[i + 1]][o]
    return hx, hy, ux, uy, L, nx, ny, eoff, et, ej, eps


DELTA = 1.0
HULL_MAX = 200_000  # targets with more samples (the ground) use the k-d search only
_EMPTY = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
          np.zeros(1, np.int64), np.zeros(0), np.zeros(0, np.int64), 0.0)


class Tree:
    """Drop-in for cKDTree(P).query(X)[0] (distances only)."""

    def __init__(self, P: np.ndarray, hull: bool | None = None):
        self.P = np.ascontiguousarray(P, dtype=np.float64)
        self.perm, self.lo, self.hi, self.box = _build(self.P, LEAF)
        self.hv, self.H = np.zeros(0, np.int64), 0
        self.hl = _EMPTY
        if (HULL["on"] if hull is None else hull) and 3 <= len(self.P) <= HULL_MAX:
            try:
                from scipy.spatial import ConvexHull
                hv = ConvexHull(self.P).vertices.astype(np.int64)
            except Exception:  # noqa: BLE001  (degenerate: collinear samples)
                hv = None
            if hv is not None and len(hv) >= 3:
                self.hv, self.H = hv, len(hv)
                self.hl = _hull_lists(self.P, hv, DELTA)

    def dist(self, X: np.ndarray) -> np.ndarray:
        return _query(np.ascontiguousarray(X, dtype=np.float64), self.P, self.perm, self.lo, self.hi, self.box, _STATE["fma"],
                      WARM, self.H, self.hv, *self.hl, DELTA)


HULL: dict = {"on": os.environ.get("TLSQ_NN_HULL", "1") == "1"}  # far-field hull path (exact; off: k-d search only)


def probe(n_targets: int = 40, seed: int = 0) -> dict:
    """Which arithmetic reproduces this platform's cKDTree bit for bit (random and boundary-like targets, near and far)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    res = {}
    for name in ("fma", "plain"):
        bad = tot = 0
        for t in range(n_targets):
            m = int(rng.integers(5, 3000))
            if t % 2:
                th = np.sort(rng.uniform(0, 2 * np.pi, m))
                P = np.c_[np.cos(th) * rng.uniform(10, 300), np.sin(th) * rng.uniform(10, 300)] + rng.uniform(-5000, 5000, 2)
            else:
                P = rng.uniform(-300, 300, (m, 2)) + rng.uniform(-5000, 5000, 2)
            X = np.cumsum(rng.normal(0, rng.choice([0.25, 30.0]), (2000, 2)), 0) + P.mean(0) + rng.uniform(-3000, 3000, 2)
            tr = Tree(P, hull=False)
            d = _query(np.ascontiguousarray(X), tr.P, tr.perm, tr.lo, tr.hi, tr.box, name == "fma", WARM, 0, tr.hv, *tr.hl, DELTA)
            ref = cKDTree(P).query(X)[0]
            bad += int((d != ref).sum())
            tot += len(X)
        res[name] = dict(mismatches=bad, queries=tot)
    return res


_LOCK = threading.Lock()


def available() -> bool:
    with _LOCK:
        return _available()


def _available() -> bool:
    if "ok" not in _STATE:
        r = probe()
        ok = [k for k in ("fma", "plain") if r[k]["mismatches"] == 0]
        _STATE.update(probe=r, ok=bool(ok), arith=ok[0] if ok else None)
        _STATE["fma"] = bool(ok) and ok[0] == "fma"
    return _STATE["ok"]
