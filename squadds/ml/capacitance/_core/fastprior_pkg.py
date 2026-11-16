"""fastprior_pkg: switchable speed components for the FROZEN physical prior of the P2d GNN (micro_mom_loc at ppg 1.5
and 3, alpha 0.49685219816506904), installed by monkeypatching the loaded frozen vendor modules (never edited).

    import fastprior_pkg as fp
    fp.install(mode="exact")            # fv + parallel + hmat_vec (+ gpu if CUDA) (+ mixed if gated)
    fp.install(mode="exact_compiled")   # exact + numba H-matrix operator (closer to the dense solve; VAL-tested, A5)
    fp.install(mode="certified")        # exact + compiled + Rayleigh-Ritz solves with an error estimate (fp.certificates())
    fp.install(components=["fv", "compiled"])   # explicit set
    fp.uninstall()

Components (protocol and gates: agents/fastprior_pkg/PROTOCOL.md; sources cited per section, code reused verbatim
where possible, only module bindings changed):
  fv        value-preserving rejection-tail fix of the windowed solve (rejtail; bit-identical)
  parallel  forked window workers (the vendor's micro_mom(workers=k)); threads per worker limited (bit-identical)
  hmat_vec  the frozen H-matrix PCG (> 12,000 panels) with a vectorised kernel, frozen tolerances (<= 1e-6)
  compiled  numba H-matrix operator for the H-matrix PCG path: closer to the exact dense solve than the frozen
            H-matrix, but up to 1.4e-5 from it (fails strict parity; opt-in, used by certified)
  gpu       dense assembly + Cholesky on CUDA (only when torch.cuda is available)
  mixed     float32 Cholesky + float64 iterative refinement (only if gated, see MIXED_GATED)
  ritz      Rayleigh-Ritz window / local solves (template start, tau 1e-3, dense <= 3,000 panels), certified
Parallel windows fork (the vendor's pool); on macOS they are always off (Accelerate is not fork-safe) and the prior
runs serially. Environment: TLSQ_PRIOR_CPUS (CPUs; default cgroup / affinity aware), TLSQ_PRIOR_WORKERS (window workers; default =
CPUs), TLSQ_PRIOR_MODE (default mode of install() without arguments).
Value hook: install(..., value_patch=fn) with fn(F, P, FL) -> undo callable (e.g. a prior-v2 patch once it passes)."""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import sys
import tempfile
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

FROZEN_DENSE_MAX = 12000  # the frozen proxy.DENSE_MAX: decides which attempts the frozen code could not solve (fv)
RITZ_CAL = 1.825367354114339  # varritz S0 calibration of the block-Jacobi estimate (s0_choice.json)
MIXED_GATED = True  # PROTOCOL.md G3 passed on Discovery x86 (A5): prior <= 4.4e-10, final matrix <= 4.5e-7
COMPONENTS = ("fv", "parallel", "hmat_vec", "compiled", "gpu", "mixed", "ritz")
CFG = dict(device="cuda", gpu_min=2000, gpu_dense_max=60000, rows=4096, ritz_tau=1e-3, ritz_maxit=12, ritz_dense_n=3000,
           ritz_start="tmpl", refine_steps=6, refine_tol=1e-13)
STATE = dict(mode=None, components=(), cpus=1, workers=1, gpu=False, value_patch=None, target=None, parallel_off=None)
F = P = FL = None  # bound vendor modules: fastprior, proxy, fastprior_loc
_ORIG: dict = {}
_UNDO: list = []
_PARENT = [0]
_CHILD = [False]
_TP: list = []
STATS = dict(compiled_calls=0)  # per-process counters
LOG: list = []  # Ritz / exact solve records of the running prior call (certified mode)
CERTS: list = []  # one certificate per micro_mom_loc call (certified mode)
_CERT_FILE = [None]


# ------------------------------------------------------------------------------------------------ binding / CPUs


def bind(target="auto"):
    """Bind the frozen vendor modules: 'stage' (tlsq.embed of the geomfast stage), 'geomgnn' (the GNN Space bundle,
    geomgnn._tlsq.embed), or a tuple (fastprior, proxy, fastprior_loc). 'auto': whichever is loaded, else stage."""
    global F, P, FL
    if isinstance(target, tuple):
        F, P, FL = target
        STATE["target"] = "modules"
        return F, P, FL
    if target == "auto":
        target = "geomgnn" if "squadds.ml.capacitance._core._tlsq.embed.fastprior" in sys.modules else "stage"
        if target == "stage" and "tlsq.embed.fastprior" not in sys.modules:
            try:
                import tlsq.embed.fastprior  # noqa: F401
            except ImportError:
                target = "geomgnn"
    if target == "geomgnn":
        from squadds.ml.capacitance._core._tlsq.embed import fastprior as F_, fastprior_loc as FL_, proxy as P_
    else:
        from tlsq.embed import fastprior as F_, fastprior_loc as FL_, proxy as P_
    F, P, FL = F_, P_, FL_
    STATE["target"] = target
    return F, P, FL


def _cgroup_cpus():
    """CPU quota of this process's cgroup (v2 cpu.max along the cgroup path, or v1 cfs quota / period); None if none."""
    best = None
    try:
        lines = open("/proc/self/cgroup").read().splitlines()
    except OSError:
        lines = []
    paths = []
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            rel = parts[2].strip("/")
            segs = rel.split("/") if rel else []
            for k in range(len(segs), -1, -1):
                paths.append(os.path.join("/sys/fs/cgroup", *segs[:k], "cpu.max"))
    paths.append("/sys/fs/cgroup/cpu.max")
    for p in paths:
        try:
            q, per = open(p).read().split()[:2]
        except (OSError, ValueError):
            continue
        if q != "max" and float(per) > 0:
            v = float(q) / float(per)
            best = v if best is None else min(best, v)
    for d in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        try:
            q = int(open(f"{d}/cpu.cfs_quota_us").read())
            per = int(open(f"{d}/cpu.cfs_period_us").read())
        except (OSError, ValueError):
            continue
        if q > 0 and per > 0:
            best = q / per if best is None else min(best, q / per)
    return best


def detect_cpus() -> int:
    """CPUs this process may use: TLSQ_PRIOR_CPUS, else min(cgroup quota, affinity, os.cpu_count())."""
    env = os.environ.get("TLSQ_PRIOR_CPUS")
    if env:
        return max(1, int(env))
    cands = [os.cpu_count() or 1]
    q = _cgroup_cpus()
    if q:
        cands.append(max(1, int(q)))
    if hasattr(os, "sched_getaffinity"):
        cands.append(len(os.sched_getaffinity(0)))
    return max(1, min(cands))


def _set_threads(n: int) -> None:
    """Limit BLAS / OpenMP (threadpoolctl), numba and the vendor's thread pool to n threads in this process."""
    try:
        from threadpoolctl import threadpool_limits
        _TP[:] = [threadpool_limits(n)]
    except Exception:  # noqa: BLE001  (no threadpoolctl / Accelerate: env vars at process start decide)
        pass
    if P is not None:
        P.NUM_THREADS = n
    try:
        import numba
        numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------------------------------------ fv
# deploy/geomgnn/src/geomgnn/rejtail_fv.py (= scripts/rejtail/rt.py policy fv), keep_last_solved=True (frozen stage),
# with the "doomed" test on the frozen DENSE_MAX (a GPU-raised DENSE_MAX must not change which attempts solve).


def _fv_solve(c, wh, o, metals, cols, eps_r):
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


def fv_window_solve(metals, n_core, h, budget, eps_r, all_rhs, levels=()):
    if all_rhs:
        return _ORIG["window_solve"](metals, n_core, h, budget, eps_r, all_rhs, levels)
    cols = np.arange(1, 1 + n_core)
    t_mesh = t_asm = t_sol = 0.0
    hh, cap, refined, majority = h, budget, 0, False
    hist = []
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
            hist.append(("doomed",) if len(c) > FROZEN_DENSE_MAX else ("mesh", c, wh, o))
            ok = False
        else:
            t = time.perf_counter()
            try:
                R, ta = _fv_solve(c, wh, o, metals, cols, eps_r)
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
        for e in reversed(hist):
            if e[0] == "R":
                R = e[1]
                break
            if e[0] == "mesh":
                try:
                    R = _fv_solve(e[1], e[2], e[3], metals, cols, eps_r)[0]
                    break
                except np.linalg.LinAlgError:
                    continue
        t_sol += time.perf_counter() - t
        assert R is not None, "no attempt solved (the frozen code raises here too)"
    return R, {"h_um": hh, "panels": len(c), "raw": n_raw, "refined": refined, "majority": majority, "mesh_s": t_mesh,
               "assemble_s": t_asm, "solve_s": t_sol}


# ------------------------------------------------------------------------------------------------ parallel windows
# The vendor's micro_mom(workers=k) (ProcessPoolExecutor, fork); windows inherit the layout through fastprior._LITE.


def _micro_mom(*args, **kw):
    ba = _ORIG["micro_mom_sig"].bind(*args, **kw)
    if sys.platform == "darwin":  # never fork on macOS, whatever the caller asked for
        ba.arguments["workers"] = 1
    if STATE["workers"] > 1 and ba.arguments.get("workers", 1) <= 1:
        ba.arguments["workers"] = STATE["workers"]
    _PARENT[0] = os.getpid()
    return _ORIG["micro_mom"](*ba.args, **ba.kwargs)


def _child_setup():
    """First window job in a forked worker: threads per worker = CPUs // workers; CUDA is never used after fork."""
    if _CHILD[0] or os.getpid() == _PARENT[0]:
        return
    _CHILD[0] = True
    nw = max(1, min(STATE["workers"], len(F._LITE.get("cores", [0]))))
    _set_threads(max(1, STATE["cpus"] // nw))
    if STATE["gpu"]:
        STATE["gpu"] = False
        P.DENSE_MAX = FROZEN_DENSE_MAX


def _lite_job(k):
    _child_setup()
    if _CHILD[0] and _CERT_FILE[0]:
        i0 = len(LOG)
        out = _ORIG["lite_job"](k)
        recs = LOG[i0:]
        for r in recs:
            r["window"] = int(k)
        with open(_CERT_FILE[0], "a") as f:
            f.write("".join(json.dumps(_jsonable(r)) + "\n" for r in recs))
        return out
    i0 = len(LOG)
    out = _ORIG["lite_job"](k)
    for r in LOG[i0:]:
        r["window"] = int(k)
    return out


def _jsonable(r):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, np.floating) else v)) for k, v in r.items()}


# ------------------------------------------------------------------------------------------------ compiled operator
# scripts/varritz/hmat.py and scripts/varritz/fastop.py (numba H-matrix of the far kernel, exact near field, FFT image
# term, block-Jacobi leaf factors on the frozen cluster tree); bodies verbatim, `P` bound late.

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # fork-safe; GNU OpenMP aborts in forked window workers
try:
    import numba as nb
    nb.config.THREADING_LAYER = os.environ["NUMBA_THREADING_LAYER"]
except ImportError:  # compiled / ritz unavailable without numba
    nb = None

if nb is not None:
    @nb.njit(fastmath=True, cache=True)
    def _krow(x, y, i, b0, b1, out):
        for j in range(b0, b1):
            dx, dy = x[i] - x[j], y[i] - y[j]
            r2 = dx * dx + dy * dy
            out[j - b0] = 1.0 / np.sqrt(r2) if r2 > 0 else 0.0

    @nb.njit(fastmath=True, cache=True)
    def _aca(x, y, a0, a1, b0, b1, eps, kmax, U, V):
        m, n = a1 - a0, b1 - b0
        used_r = np.zeros(m, np.bool_)
        used_c = np.zeros(n, np.bool_)
        r = np.empty(n)
        u = np.empty(m)
        norm2 = 0.0
        i = 0
        k = 0
        while k < min(m, n, kmax):
            _krow(x, y, a0 + i, b0, b1, r)
            for q in range(k):
                ui = U[i, q]
                for j in range(n):
                    r[j] -= ui * V[j, q]
            used_r[i] = True
            jbest, vbest = -1, 0.0
            for j in range(n):
                if not used_c[j] and abs(r[j]) > vbest:
                    vbest, jbest = abs(r[j]), j
            if jbest < 0 or vbest == 0.0:
                return k
            piv = r[jbest]
            for j in range(n):
                V[j, k] = r[j] / piv
            _krow(x, y, b0 + jbest, a0, a1, u)
            for q in range(k):
                vj = V[jbest, q]
                for ii in range(m):
                    u[ii] -= vj * U[ii, q]
            for ii in range(m):
                U[ii, k] = u[ii]
            used_c[jbest] = True
            nu, nv = 0.0, 0.0
            for ii in range(m):
                nu += u[ii] * u[ii]
            for j in range(n):
                nv += V[j, k] * V[j, k]
            cross = 0.0
            for q in range(k):
                su, sv = 0.0, 0.0
                for ii in range(m):
                    su += u[ii] * U[ii, q]
                for j in range(n):
                    sv += V[j, k] * V[j, q]
                cross += su * sv
            norm2 += nu * nv + 2.0 * cross
            k += 1
            if nu * nv <= eps * eps * norm2:
                return k
            ibest, ub = -1, 0.0
            for ii in range(m):
                if not used_r[ii] and abs(u[ii]) > ub:
                    ub, ibest = abs(u[ii]), ii
            if ibest < 0:
                return k
            i = ibest
        return -1 if k >= kmax and k < min(m, n) else k

    @nb.njit(parallel=True, cache=True)
    def _build(x, y, lows, eps, kmax, Ubuf, Vbuf, uoff, voff, ranks, dense, doff, ddata):
        nl = lows.shape[0]
        for b in nb.prange(nl):
            a0, a1, b0, b1 = lows[b, 0], lows[b, 1], lows[b, 2], lows[b, 3]
            m, n = a1 - a0, b1 - b0
            U = Ubuf[uoff[b]:uoff[b] + m * kmax].reshape(m, kmax)
            V = Vbuf[voff[b]:voff[b] + n * kmax].reshape(n, kmax)
            ranks[b] = _aca(x, y, a0, a1, b0, b1, eps, kmax, U, V)
        nd = dense.shape[0]
        for b in nb.prange(nd):
            a0, a1, b0, b1 = dense[b, 0], dense[b, 1], dense[b, 2], dense[b, 3]
            n = b1 - b0
            row = np.empty(n)
            for i in range(a0, a1):
                _krow(x, y, i, b0, b1, row)
                ddata[doff[b] + (i - a0) * n:doff[b] + (i - a0 + 1) * n] = row

    @nb.njit(parallel=True, cache=True)
    def _apply(X, lows, kmax, Ubuf, Vbuf, uoff, voff, ranks, dense, doff, ddata, nthr, Y):
        n, mc = X.shape
        buf = np.zeros((nthr, n, mc))
        nl = lows.shape[0]
        nd = dense.shape[0]
        tot = nl + nd
        for t in nb.prange(nthr):
            B = buf[t]
            for b in range(t, tot, nthr):
                if b < nl:
                    a0, a1, b0, b1 = lows[b, 0], lows[b, 1], lows[b, 2], lows[b, 3]
                    m, nn = a1 - a0, b1 - b0
                    k = ranks[b]
                    U = Ubuf[uoff[b]:uoff[b] + m * kmax].reshape(m, kmax)[:, :k]
                    V = Vbuf[voff[b]:voff[b] + nn * kmax].reshape(nn, kmax)[:, :k]
                    B[a0:a1] += U @ (V.T @ X[b0:b1])
                    B[b0:b1] += V @ (U.T @ X[a0:a1])
                else:
                    q = b - nl
                    a0, a1, b0, b1 = dense[q, 0], dense[q, 1], dense[q, 2], dense[q, 3]
                    D = ddata[doff[q]:doff[q] + (a1 - a0) * (b1 - b0)].reshape(a1 - a0, b1 - b0)
                    B[a0:a1] += D @ X[b0:b1]
                    if a0 != b0:
                        B[b0:b1] += D.T @ X[a0:a1]
        for i in nb.prange(n):
            for c in range(mc):
                s = 0.0
                for t in range(nthr):
                    s += buf[t, i, c]
                Y[i, c] = s

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _far(x, y, Xt, Y, T=64):
        m, n = Xt.shape
        nt = (n + T - 1) // T
        for it in nb.prange(nt):
            r = np.empty(n)
            for i in range(it * T, min(n, it * T + T)):
                xi, yi = x[i], y[i]
                for j in range(n):
                    dx, dy = xi - x[j], yi - y[j]
                    r[j] = dx * dx + dy * dy
                r[i] = 1.0
                for j in range(n):
                    r[j] = 1.0 / np.sqrt(r[j])
                r[i] = 0.0
                for k in range(m):
                    s = 0.0
                    for j in range(n):
                        s += r[j] * Xt[k, j]
                    Y[i, k] = s


def _numba():
    if nb is None:
        raise ImportError("numba is required for the compiled / ritz components")
    return dict(nb=nb, build=_build, apply=_apply, far=_far)


class HFar:
    """Far kernel sum_{j != i} x_j / d_ij as an H-matrix (frozen block plan: leaf 128, eta 2). (varritz/hmat.py)"""

    def __init__(self, c, leaf=128, eta=2.0, eps=1e-7, kmax=48, nthr=None):
        K = _numba()
        self.perm, nodes = P._cluster_tree(c, leaf)
        self.x, self.y = np.ascontiguousarray(c[self.perm, 0]), np.ascontiguousarray(c[self.perm, 1])
        plan_l, plan_d = [], []

        def leafq(k):
            return nodes[k][4] < 0

        def rec(s, t):
            a0, a1, lo_s, hi_s, sl, sr = nodes[s]
            b0, b1, lo_t, hi_t, tl, tr = nodes[t]
            if s == t:
                if leafq(s):
                    plan_d.append((a0, a1, a0, a1))
                else:
                    rec(sl, sl), rec(sr, sr), rec(sl, sr)
                return
            gap = np.maximum(0.0, np.maximum(lo_s - hi_t, lo_t - hi_s))
            diam = min(np.sqrt(((hi_s - lo_s) ** 2).sum()), np.sqrt(((hi_t - lo_t) ** 2).sum()))
            if diam <= eta * np.sqrt((gap**2).sum()):
                plan_l.append((a0, a1, b0, b1))
            elif leafq(s) and leafq(t):
                plan_d.append((a0, a1, b0, b1))
            elif leafq(s):
                rec(s, tl), rec(s, tr)
            elif leafq(t):
                rec(sl, t), rec(sr, t)
            else:
                for u in (sl, sr):
                    for v in (tl, tr):
                        rec(u, v)

        rec(0, 0)
        lows = np.array(plan_l, np.int64).reshape(-1, 4)
        dense = np.array(plan_d, np.int64).reshape(-1, 4)
        m = lows[:, 1] - lows[:, 0]
        nn = lows[:, 3] - lows[:, 2]
        uoff = np.r_[0, np.cumsum(m * kmax)][:-1]
        voff = np.r_[0, np.cumsum(nn * kmax)][:-1]
        Ubuf, Vbuf = np.zeros(int((m * kmax).sum())), np.zeros(int((nn * kmax).sum()))
        dsz = (dense[:, 1] - dense[:, 0]) * (dense[:, 3] - dense[:, 2])
        doff = np.r_[0, np.cumsum(dsz)][:-1]
        ddata = np.zeros(int(dsz.sum()))
        ranks = np.zeros(len(lows), np.int64)
        K["build"](self.x, self.y, lows, eps, kmax, Ubuf, Vbuf, uoff, voff, ranks, dense, doff, ddata)
        bad = ranks < 0
        if bad.any():  # rank above kmax: these blocks become dense
            extra = lows[bad]
            lows = lows[~bad]
            keep_u, keep_v = uoff[~bad], voff[~bad]
            ranks = ranks[~bad]
            uoff, voff = keep_u, keep_v
            esz = (extra[:, 1] - extra[:, 0]) * (extra[:, 3] - extra[:, 2])
            eoff = np.r_[0, np.cumsum(esz)][:-1]
            edata = np.zeros(int(esz.sum()))
            K["build"](self.x, self.y, np.zeros((0, 4), np.int64), eps, kmax, Ubuf, Vbuf, uoff[:0], voff[:0], ranks[:0], extra, eoff, edata)
            doff = np.r_[doff, eoff + len(ddata)]
            ddata = np.r_[ddata, edata]
            dense = np.r_[dense, extra]
        self.args = (lows, kmax, Ubuf, Vbuf, uoff, voff, ranks, dense, doff, ddata)
        self.nthr = nthr or K["nb"].get_num_threads()
        self.n_low, self.n_dense, self.mean_rank = len(lows), len(dense), float(ranks.mean()) if len(ranks) else 0.0
        self.n_converted = int(bad.sum())

    def __call__(self, X):
        Xp = np.ascontiguousarray(X[self.perm])
        Yp = np.empty_like(Xp)
        _numba()["apply"](Xp, *self.args, self.nthr, Yp)
        Y = np.empty_like(Yp)
        Y[self.perm] = Yp
        return Y


def _lagrange(u, q):
    i0 = np.floor(u).astype(np.int64) - (q // 2 - 1)
    t = u - i0
    w = np.ones((len(u), q))
    for a in range(q):
        for b in range(q):
            if a != b:
                w[:, a] *= (t - b) / (a - b)
    return i0, w


class ImageConv:
    """Smooth finite-substrate image kernel (proxy._image_correction, exact series) by grid convolution. (fastop.py)"""

    def __init__(self, c, eps_r, spacing_frac=1 / 16, q=6):
        from scipy.sparse import csr_matrix
        P._image_correction(np.zeros(1), eps_r)
        key = (P.SUBSTRATE["model"], float(P.SUBSTRATE["t_um"]), float(eps_r))
        _, _, w, D = P._IMG_CACHE[key]
        self.w, self.D = w, D
        h = spacing_frac * D.min()
        lo = c.min(0) - (q // 2 + 1) * h
        u = (c - lo) / h
        ix0, wx = _lagrange(u[:, 0], q)
        iy0, wy = _lagrange(u[:, 1], q)
        nx, ny = int(ix0.max()) + q + 1, int(iy0.max()) + q + 1
        rows = np.repeat(np.arange(len(c)), q * q)
        gx = (ix0[:, None, None] + np.arange(q)[None, :, None]) * np.ones((1, 1, q), np.int64)
        gy = (iy0[:, None, None] + np.arange(q)[None, None, :]) * np.ones((1, q, 1), np.int64)
        vals = (wx[:, :, None] * wy[:, None, :]).ravel()
        self.W = csr_matrix((vals, (rows, (gx * ny + gy).ravel())), shape=(len(c), nx * ny))
        self.nx, self.ny = nx, ny
        Lx, Ly = 2 * nx, 2 * ny
        ax = np.minimum(np.arange(Lx), Lx - np.arange(Lx)) * h
        ay = np.minimum(np.arange(Ly), Ly - np.arange(Ly)) * h
        rho = np.sqrt(ax[:, None] ** 2 + ay[None, :] ** 2)
        Kk = sum(wn / np.sqrt(rho**2 + Dn * Dn) for wn, Dn in zip(w, D))
        self.Kf = np.fft.rfft2(Kk)
        self.L = (Lx, Ly)

    def __call__(self, X):
        G = (self.W.T @ X).reshape(self.nx, self.ny, -1)
        out = np.empty((self.nx * self.ny, X.shape[1]))
        for k in range(X.shape[1]):
            f = np.fft.irfft2(np.fft.rfft2(G[:, :, k], s=self.L) * self.Kf, s=self.L)[: self.nx, : self.ny]
            out[:, k] = f.ravel()
        return self.W @ out


class FastOp:
    """The frozen panel operator, matrix-free; bj(): block-Jacobi on the frozen cluster tree's leaves. (fastop.py)"""

    def __init__(self, c, wh, o, eps_r, M, leaf=128, h_min_n=4000, h_eps=1e-8):
        from scipy.sparse import csr_matrix
        t = time.perf_counter()
        self.H = None
        if len(c) >= h_min_n:
            self.H = HFar(c, eps=h_eps)
        self.t_hmat = time.perf_counter() - t
        t = time.perf_counter()
        self.c, self.wh, self.o, self.n, self.M = c, wh, o, len(c), M
        self.B = (o[:, None] == np.arange(M)[None]).astype(float)
        self.k = 4 * np.pi * P.EPS0_FF_PER_UM * (eps_r + 1) / 2
        self.x, self.y = np.ascontiguousarray(c[:, 0]), np.ascontiguousarray(c[:, 1])
        ii, jj = P._near_pairs(c, wh.max(1))
        d = np.sqrt(((c[ii] - c[jj]) ** 2).sum(1))
        near = 0.5 * (P.rect_potential(c[ii], c[jj], wh[jj]) + P.rect_potential(c[jj], c[ii], wh[ii])) - 1.0 / d
        dg = P.rect_self(wh[:, 0], wh[:, 1])
        n = self.n
        self.N = csr_matrix((np.r_[near, near, dg], (np.r_[ii, jj, np.arange(n)], np.r_[jj, ii, np.arange(n)])), shape=(n, n))
        self.t_near = time.perf_counter() - t
        t = time.perf_counter()
        self.img = ImageConv(c, eps_r) if P._image_correction(np.zeros(1), eps_r) is not None else None
        self.t_img = time.perf_counter() - t
        t = time.perf_counter()
        perm, nodes = P._cluster_tree(c, leaf)
        self.leaves = []
        img0 = P._image_correction
        for a0, a1, lo, hi, l, r in nodes:
            if l >= 0:
                continue
            ix = np.sort(perm[a0:a1])
            cc = c[ix]
            dd = np.sqrt(((cc[:, None] - cc[None]) ** 2).sum(-1))
            np.fill_diagonal(dd, 1.0)
            A = 1.0 / dd
            np.fill_diagonal(A, 0.0)
            A += self.N[ix][:, ix].toarray()
            if self.img is not None:
                np.fill_diagonal(dd, 0.0)
                A += img0(dd, eps_r, exact=True)
            self.leaves.append((ix, cho_factor(A / self.k, lower=True)))
        self.t_blocks = time.perf_counter() - t

    def matvec(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if self.H is not None:
            Y = self.H(X)
        else:
            Y = np.empty_like(X)
            _numba()["far"](self.x, self.y, np.ascontiguousarray(X.T), Y)
        Y += self.N @ X
        if self.img is not None:
            Y += self.img(X)
        return Y / self.k

    def bj(self, R):
        Z = np.zeros_like(R)
        for ix, cf in self.leaves:
            Z[ix] = cho_solve(cf, R[ix], check_finite=False)
        return Z


def compiled_pcg(c, wh, o, n_metals, eps_r):
    """Panel charges (n, n_metals) by the frozen block-Jacobi PCG (proxy._HMatrix.pcg: all columns, stop when every
    column's relative residual < HMAT tol, LinAlgError after maxit) on the compiled operator. A zero right-hand-side
    column (a metal without panels) gives residual 0/0 = NaN forever, so it raises at once instead of after maxit."""
    STATS["compiled_calls"] += 1
    B = (o[:, None] == np.arange(n_metals)[None, :]).astype(float)
    bn = np.sqrt((B * B).sum(0))
    if not np.all(bn > 0):
        raise np.linalg.LinAlgError("H-matrix PCG: a metal without panels (zero right-hand side; frozen: NaN residual)")
    op = FastOp(c, wh, o, eps_r, n_metals, h_eps=P.HMAT["eps"])
    tol, maxit = P.HMAT["tol"], P.HMAT["maxit"]
    X, R = np.zeros_like(B), B.copy()
    Z = op.bj(R)
    Pd, rz = Z.copy(), (R * Z).sum(0)
    res = np.inf
    for it in range(maxit):
        AP = op.matvec(Pd)
        al = rz / (Pd * AP).sum(0)
        X += al * Pd
        R -= al * AP
        res = (np.sqrt((R * R).sum(0)) / bn).max()
        if res < tol:
            return X
        if not np.isfinite(res):
            break
        Z = op.bj(R)
        rzn = (R * Z).sum(0)
        Pd = Z + (rzn / rz) * Pd
        rz = rzn
    raise np.linalg.LinAlgError(f"compiled H-matrix PCG did not converge (residual {res:.1e})")


def _mom_solve_fast(c, wh, o, n_metals, eps_r=11.45, dense_max=None):
    n = len(c)
    if n <= (P.DENSE_MAX if dense_max is None else dense_max) or "compiled" not in STATE["components"]:
        return _ORIG["mom_solve_fast"](c, wh, o, n_metals, eps_r, dense_max)
    return compiled_pcg(c, wh, o, n_metals, eps_r)


# ------------------------------------------------------------------------------------------------ hmat_vec (fastmom.py)
# scripts/geomfast/fastmom.py: the frozen H-matrix (same cluster tree, ACA, block-Jacobi PCG, FROZEN tolerances) with a
# vectorised kernel (no per-row scipy sparse slicing, no Python loop over image charges or ACA ranks). Only the H-matrix
# kernel is patched (proxy._image_correction and the dense path are untouched).


def _hv_images(eps_r):
    P._image_correction(np.zeros(1), eps_r)
    key = (P.SUBSTRATE["model"], float(P.SUBSTRATE["t_um"]), float(eps_r))
    w, D = P._IMG_CACHE[key][2:]
    return np.asarray(w, float), np.asarray(D, float) ** 2


def _hv_kernel_init(self, c, wh, eps_r, perm):
    _ORIG["Kernel.__init__"](self, c, wh, eps_r, perm)
    N = self.N.tocsr()
    N.sort_indices()
    self.indptr, self.indices, self.data = N.indptr, N.indices, N.data
    self.w, self.D2 = _hv_images(eps_r) if self.img else (None, None)


def _hv_corr(self, d):
    return np.sqrt((d * d)[..., None] + self.D2) ** -1 @ self.w


def _hv_row(self, i, b0, b1):
    d = np.sqrt((self.x[i] - self.x[b0:b1]) ** 2 + (self.y[i] - self.y[b0:b1]) ** 2)
    r = 1.0 / d
    if self.img:
        r += _hv_corr(self, d)
    s, e = self.indptr[i], self.indptr[i + 1]
    if e > s:
        ix = self.indices[s:e]
        lo, hi = np.searchsorted(ix, (b0, b1))
        if hi > lo:
            r[ix[lo:hi] - b0] += self.data[s + lo:s + hi]
    return r / self.k


def _hv_block(self, a0, a1, b0, b1):
    d = np.sqrt((self.x[a0:a1, None] - self.x[None, b0:b1]) ** 2 + (self.y[a0:a1, None] - self.y[None, b0:b1]) ** 2)
    corr = _hv_corr(self, d) if self.img else None
    if a0 == b0:
        d[np.arange(a1 - a0), np.arange(a1 - a0)] = 1.0
    A = 1.0 / d
    if corr is not None:
        A += corr
    s, e = self.indptr[a0], self.indptr[a1]
    if e > s:
        ix = self.indices[s:e]
        rr = np.repeat(np.arange(a1 - a0), np.diff(self.indptr[a0:a1 + 1]))
        m = (ix >= b0) & (ix < b1)
        np.add.at(A, (rr[m], ix[m] - b0), self.data[s:e][m])
    return A / self.k


def _hv_aca(K, a0, a1, b0, b1, eps, max_rank):
    m, n = a1 - a0, b1 - b0
    R = min(m, n, max_rank)
    U, V = np.empty((m, R)), np.empty((n, R))
    used_r, used_c = np.zeros(m, bool), np.zeros(n, bool)
    i, norm2, k = 0, 0.0, 0
    for _ in range(R):
        r = K.row(a0 + i, b0, b1)
        if k:
            r -= V[:, :k] @ U[i, :k]
        rm = np.where(used_c, 0.0, np.abs(r))
        j = int(np.argmax(rm))
        if rm[j] == 0.0:
            break
        v = r / r[j]
        u = K.row(b0 + j, a0, a1)
        if k:
            u -= U[:, :k] @ V[j, :k]
        used_r[i], used_c[j] = True, True
        nu, nv = float(u @ u), float(v @ v)
        norm2 += nu * nv + (2 * float((U[:, :k].T @ u) @ (V[:, :k].T @ v)) if k else 0.0)
        U[:, k], V[:, k] = u, v
        k += 1
        if nu * nv <= eps**2 * norm2:
            break
        um = np.where(used_r, 0.0, np.abs(u))
        i = int(np.argmax(um))
        if um[i] == 0.0:
            break
    else:
        return None
    return U[:, :k].copy(), V[:, :k].copy()


def _hv_mom_solve_fast(c, wh, o, n_metals, eps_r=11.45, dense_max=None):
    """The frozen mom_solve_fast (vectorised kernel); a zero right-hand-side column raises at once (frozen: after maxit)."""
    if len(c) > (P.DENSE_MAX if dense_max is None else dense_max) and not np.all(np.bincount(o, minlength=n_metals) > 0):
        raise np.linalg.LinAlgError("H-matrix PCG: a metal without panels (zero right-hand side; frozen: NaN residual)")
    return _ORIG["mom_solve_fast"](c, wh, o, n_metals, eps_r, dense_max)


# ------------------------------------------------------------------------------------------------ gpu (fastgpu.py)


class _Dev:
    def __init__(self, t):
        self.t = t


def _gpu_assemble(c, wh, diag, eps_r, near):
    n = len(c)
    if not STATE["gpu"] or n < CFG["gpu_min"]:
        return _ORIG["assemble"](c, wh, diag, eps_r, near)
    import torch
    dev = CFG["device"]
    k = 4 * np.pi * P.EPS0_FF_PER_UM * (eps_r + 1) / 2
    img = P._image_correction(np.zeros(1), eps_r) is not None
    if img:
        key = (P.SUBSTRATE["model"], float(P.SUBSTRATE["t_um"]), float(eps_r))
        du, T = P._IMG_CACHE[key][:2]
        T = torch.as_tensor(T, dtype=torch.float64, device=dev)
    x = torch.as_tensor(c[:, 0], dtype=torch.float64, device=dev)
    y = torch.as_tensor(c[:, 1], dtype=torch.float64, device=dev)
    A = torch.empty((n, n), dtype=torch.float64, device=dev)
    for r0 in range(0, n, CFG["rows"]):
        r1 = min(r0 + CFG["rows"], n)
        d = torch.sqrt((x[r0:r1, None] - x[None, :]) ** 2 + (y[r0:r1, None] - y[None, :]) ** 2)
        if img:
            u = d / du
            j = torch.clamp(u.floor().long(), 0, len(T) - 2)
            f = u - j
            corr = T[j] + f * (T[j + 1] - T[j])
        d[torch.arange(r1 - r0, device=dev), torch.arange(r0, r1, device=dev)] = 1.0
        blk = 1.0 / d
        if img:
            blk += corr
        A[r0:r1] = blk / k
    ii, jj = P._near_pairs(c, wh.max(1))
    if len(ii):
        v = 0.5 * (near(c[ii], c[jj], wh[jj]) + near(c[jj], c[ii], wh[ii])) / k
        if img:
            v = v + P._image_correction(np.sqrt(((c[ii] - c[jj]) ** 2).sum(1)), eps_r) / k
        vi = torch.as_tensor(v, dtype=torch.float64, device=dev)
        ti, tj = torch.as_tensor(ii, device=dev), torch.as_tensor(jj, device=dev)
        A[ti, tj] = vi
        A[tj, ti] = vi
    dg = diag / k if not img else (diag + P._image_correction(np.zeros(1), eps_r)[0]) / k
    A[torch.arange(n, device=dev), torch.arange(n, device=dev)] = torch.as_tensor(dg, dtype=torch.float64, device=dev)
    return _Dev(A)


def _gpu_chol(A, B):
    import torch
    L, info = torch.linalg.cholesky_ex(A.t)
    if int(info) > 0:
        raise np.linalg.LinAlgError("not positive definite (device Cholesky)")
    Bt = torch.as_tensor(np.ascontiguousarray(B), dtype=torch.float64, device=A.t.device)
    X = torch.cholesky_solve(Bt, L).cpu().numpy()
    A.t = None  # the matrix is not used after a successful solve: release it and the allocator cache (shared GPU)
    del L, Bt
    torch.cuda.empty_cache()
    return X


# ------------------------------------------------------------------------------------------------ mixed (fastmom.py)


def _mixed_chol(A, B):
    from scipy.linalg.blas import dsymm
    try:
        cf = cho_factor(A.astype(np.float32, order="F"), lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        return _ORIG["chol_solve"](A.copy(order="F"), B)
    B = np.asfortranarray(B, dtype=float)
    X = cho_solve(cf, B.astype(np.float32), check_finite=False).astype(float)
    bn = np.linalg.norm(B, axis=0)
    for _ in range(CFG["refine_steps"]):
        R = B - dsymm(1.0, A, X, side=0, lower=1)
        if (np.linalg.norm(R, axis=0) <= CFG["refine_tol"] * bn).all():
            return X
        X += cho_solve(cf, R.astype(np.float32), check_finite=False)
    return _ORIG["chol_solve"](A.copy(order="F"), B)


def _chol_solve(A, B):
    if isinstance(A, _Dev):
        return _gpu_chol(A, B)
    if "mixed" in STATE["components"]:
        return _mixed_chol(A, B)
    return _ORIG["chol_solve"](A, B)


def _sym_solve(A, B):
    if not isinstance(A, _Dev):
        if "mixed" not in STATE["components"]:
            return _ORIG["sym_solve"](A, B)
        try:  # = _ORIG["sym_solve"] without its copy of A: _mixed_chol only reads A (its float64 fallbacks copy it)
            return _mixed_chol(A, B)
        except np.linalg.LinAlgError:
            from scipy.linalg import solve
            return solve(A, B, lower=True, assume_a="sym", overwrite_a=True, check_finite=False)
    if A.t.shape[0] > FROZEN_DENSE_MAX:
        return _gpu_chol(A, B)
    try:
        return _gpu_chol(A, B)
    except np.linalg.LinAlgError:
        from scipy.linalg import solve
        return solve(A.t.cpu().numpy(), B, lower=True, assume_a="sym", overwrite_a=True, check_finite=False)


# ------------------------------------------------------------------------------------------------ ritz (certified)
# scripts/varritz/vr.py (templates, Ritz) and scripts/varritz/ritzprior.py (window / local loops, A5 dense dispatch).


def edge_geometry(metals, c, wh, o):
    import shapely
    n, M = len(c), len(metals)
    d, a, g = np.zeros(n), np.zeros(n), np.zeros(n)
    D = np.zeros((n, M))
    pts = shapely.points(c)
    for k in range(M):
        ix = np.nonzero(o == k)[0]
        if not len(ix):
            continue
        bd = metals[k].boundary
        shapely.prepare(bd)
        ln = shapely.shortest_line(pts[ix], bd)
        xy = shapely.get_coordinates(ln).reshape(-1, 2, 2)
        v = xy[:, 0] - xy[:, 1]
        dk = np.sqrt((v**2).sum(1))
        nv = np.where(dk[:, None] > 0, v / np.maximum(dk, 1e-300)[:, None], 0.7071)
        d[ix], a[ix] = dk, np.abs(nv[:, 0]) * wh[ix, 0] + np.abs(nv[:, 1]) * wh[ix, 1]
        others = shapely.union_all([metals[j] for j in range(M) if j != k])
        shapely.prepare(others)
        g[ix] = shapely.distance(shapely.points(xy[:, 1]), others)
    for j in range(M):
        shapely.prepare(metals[j])
        D[:, j] = shapely.distance(pts, metals[j])
    return d, a, np.maximum(g, 1e-9), D


def templates(d, a, g):
    lo, hi = np.maximum(d - a / 2, 0.0), d + a / 2
    w = np.maximum(hi - lo, 1e-12)
    s = 2 * (np.sqrt(hi) - np.sqrt(lo)) / w
    gg = 2 * (np.arcsinh(np.sqrt(hi / g)) - np.arcsinh(np.sqrt(lo / g))) / w
    return {"u": np.ones_like(d), "s": s, "g": gg}


def _orth_blocks(V, o, M):
    out = []
    for k in range(M):
        ix = np.nonzero(o == k)[0]
        cols = np.nonzero(np.abs(V[ix]).sum(0) > 0)[0]
        if not len(cols):
            continue
        Vk = V[np.ix_(ix, cols)]
        Vk = Vk / np.maximum(np.linalg.norm(Vk, axis=0), 1e-300)
        U, s, _ = np.linalg.svd(Vk, full_matrices=False)
        keep = s > 1e-10 * s[0]
        W = np.zeros((len(o), int(keep.sum())))
        W[ix] = U[:, keep]
        out.append(W)
    return np.hstack(out)


def ritz(V, PV, B):
    G = V.T @ PV
    G = 0.5 * (G + G.T)
    Fm = V.T @ B
    try:
        Y = cho_solve(cho_factor(G, lower=True), Fm)
    except np.linalg.LinAlgError:
        lam, U = eigh(G)
        keep = lam > 1e-13 * lam.max()
        Y = U[:, keep] @ ((U[:, keep].T @ Fm) / lam[keep, None])
    CV = Fm.T @ Y
    return 0.5 * (CV + CV.T), Y, B - PV @ Y


def start_basis(metals, c, wh, o, kind, op, B):
    M = len(metals)
    if kind == "pcg":
        Z = op.bj(B)
        return Z / np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
    area = wh[:, 0] * wh[:, 1]
    d, a, g, _ = edge_geometry(metals, c, wh, o)
    T = templates(d, a, g)
    return _orth_blocks(np.hstack([np.where((o == k)[:, None], (T[nm] * area)[:, None], 0.0)
                                   for k in range(M) for nm in "usg"]), o, M)


def criterion(CV, est, floor=0.1):
    e = np.sqrt(np.maximum(np.diag(est), 0.0))
    return float(np.max(np.outer(e, e) / np.maximum(np.abs(CV), floor)))


def ritz_solve(metals, c, wh, o, eps_r, cols):
    tau, start, maxit = CFG["ritz_tau"], CFG["ritz_start"], CFG["ritz_maxit"]
    t0 = time.perf_counter()
    op = FastOp(c, wh, o, eps_r, len(metals))
    B = op.B
    V = start_basis(metals, c, wh, o, start, op, B)
    PV = op.matvec(V)
    for m in range(maxit + 1):
        CV, Y, R = ritz(V, PV, B)
        est = R.T @ op.bj(R)
        rel = criterion(CV, est)
        if rel <= tau or m == maxit:
            break
        Z = op.bj(R)
        Z /= np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
        V, PV = np.hstack([V, Z]), np.hstack([PV, op.matvec(Z)])
    info = dict(n=len(c), steps=m, converged=rel <= tau, est_rel=rel, est_diag=np.diag(est).tolist(), k=V.shape[1],
                t=time.perf_counter() - t0)
    return (V @ Y)[:, np.asarray(cols)], info


def _frozen_solve(c, wh, o, eps_r, cols, all_rhs, n_metals):
    if len(c) <= P.DENSE_MAX:
        A = P._assemble(c, wh, P.rect_self(wh[:, 0], wh[:, 1]), eps_r, P.rect_potential)
        Bm = (o[:, None] == cols[None, :]).astype(float)
        return P._chol_solve(A, Bm) if all_rhs else F._sym_solve(A, Bm)
    return P.mom_solve_fast(c, wh, o, n_metals, eps_r)[:, cols]


def _ritz_dense_n():
    return max(CFG["ritz_dense_n"], CFG["gpu_dense_max"] if STATE["gpu"] else 0)


def _solve_mesh(metals, c, wh, o, eps_r, cols, all_rhs, kind):
    if len(c) <= _ritz_dense_n():  # A5 (and the GPU): exact solve, estimate 0
        Q = _frozen_solve(c, wh, o, eps_r, cols, all_rhs, len(metals))
        LOG.append(dict(kind=kind, ritz=None, exact=True, steps=0, n=len(c), est_rel=0.0, est_diag=[0.0] * len(metals)))
        return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)
    try:
        Q, info = ritz_solve(metals, c, wh, o, eps_r, cols)
        rec = dict(kind=kind, ritz=bool(info["converged"]), exact=False, **info)
        if not info["converged"]:
            raise np.linalg.LinAlgError("ritz not converged")
    except np.linalg.LinAlgError:
        LOG.append(dict(kind=kind, ritz=False, exact=True, fallback=True, steps=0, n=len(c), est_rel=0.0, est_diag=[0.0] * len(metals)))
        Q = _frozen_solve(c, wh, o, eps_r, cols, all_rhs, len(metals))
        return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)
    LOG.append(rec)
    return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)


def _last_good(R, ok, skipped, metals, eps_r, cols, all_rhs):
    if R is not None and (ok or not np.isnan(R).any()):
        return R
    for c, wh, o in reversed(skipped):
        try:
            Q = _frozen_solve(c, wh, o, eps_r, cols, all_rhs, len(metals))
            LOG.append(dict(kind="last_good", ritz=None, exact=True, steps=0, n=len(c), est_rel=0.0, est_diag=[0.0] * len(metals)))
            return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("no successful solve in the refinement sequence")


def _final_unpanelled(metals, c, wh, o, eps_r, cols, all_rhs):
    """Final majority mesh still missing a metal: the frozen solve (doomed above the frozen DENSE_MAX: it raises)."""
    if len(c) > FROZEN_DENSE_MAX:
        raise np.linalg.LinAlgError("final mesh misses a metal above DENSE_MAX (frozen: NaN PCG)")
    Q = _frozen_solve(c, wh, o, eps_r, cols, all_rhs, len(metals))
    LOG.append(dict(kind="window", ritz=None, exact=True, steps=0, n=len(c), est_rel=0.0, est_diag=[0.0] * len(metals)))
    return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)


def ritz_window_solve(metals, n_core, h, budget, eps_r, all_rhs, levels=()):
    cols = np.arange(len(metals)) if all_rhs else np.arange(1, 1 + n_core)
    t_mesh = t_sol = 0.0
    hh, cap, refined, majority = h, budget, 0, False
    skipped, R = [], None
    while True:
        t = time.perf_counter()
        if not refined:
            c, wh, o, hh, n_raw = F._lite_mesh(metals, hh, cap, n_core, levels)
        elif levels:
            c, wh, o, hh, n_raw = F._graded_mesh(metals, n_core, hh, levels, majority)
        else:
            c, wh, o, hh, n_raw = P.mesh_flagship(metals, hh, cap, majority=majority)
        t_mesh += time.perf_counter() - t
        t = time.perf_counter()
        full = bool(np.all(np.bincount(o, minlength=len(metals)) > 0))
        if not full and not majority:
            skipped.append((c, wh, o))
            if refined < F.LITE_REFINE:
                hh, cap, refined = hh / 2, np.inf, refined + 1
            else:
                majority = True
            continue
        try:
            if not full:
                R = _final_unpanelled(metals, c, wh, o, eps_r, cols, all_rhs)
            else:
                R = _solve_mesh(metals, c, wh, o, eps_r, cols, all_rhs, "window")
            off = np.ones(R.shape, bool)
            off[np.arange(len(cols)), cols] = False
            ok = bool(np.all(np.bincount(o, minlength=len(metals)) > 0) and (not all_rhs or np.all(R[off] < 0)))
        except np.linalg.LinAlgError:
            ok = False
        t_sol += time.perf_counter() - t
        if LOG:
            LOG[-1].update(accepted=ok, refined=refined)
        if ok or majority:
            break
        if refined < F.LITE_REFINE:
            hh, cap, refined = hh / 2, np.inf, refined + 1
        else:
            majority = True
    R = _last_good(R, ok, skipped, metals, eps_r, cols, all_rhs)
    return R, {"h_um": hh, "panels": len(c), "raw": n_raw, "refined": refined, "majority": majority, "mesh_s": t_mesh,
               "assemble_s": 0.0, "solve_s": t_sol}


def ritz_loc_solve(metals, H, h_floor, ppg, eps_r):
    t_mesh = t_sol = 0.0
    cols = np.arange(len(metals))
    refined, majority = 0, False
    skipped, R = [], None
    while True:
        t = time.perf_counter()
        c, wh, o, n_raw = FL.mesh_local(metals, H, h_floor, ppg, majority)
        t_mesh += time.perf_counter() - t
        t = time.perf_counter()
        full = bool(np.all(np.bincount(o, minlength=len(metals)) > 0))
        if not full and not majority:
            skipped.append((c, wh, o))
            if refined < F.LITE_REFINE:
                H, h_floor, refined = H / 2, h_floor / 2, refined + 1
            else:
                majority = True
            continue
        try:
            if not full:
                R = _final_unpanelled(metals, c, wh, o, eps_r, cols, True)
            else:
                R = _solve_mesh(metals, c, wh, o, eps_r, cols, True, "local")
            off = ~np.eye(len(metals), dtype=bool)
            ok = bool(np.all(np.bincount(o, minlength=len(metals)) > 0) and np.all(R[off] < 0))
        except np.linalg.LinAlgError:
            ok = False
        t_sol += time.perf_counter() - t
        if LOG:
            LOG[-1].update(accepted=ok, refined=refined)
        if ok or majority:
            break
        if refined < F.LITE_REFINE:
            H, h_floor, refined = H / 2, h_floor / 2, refined + 1
        else:
            majority = True
    R = _last_good(R, ok, skipped, metals, eps_r, cols, True)
    return R, {"H_um": H, "h_floor_um": h_floor, "h_min_um": float(wh.min()), "panels": len(c), "raw": n_raw,
               "refined": refined, "majority": majority, "mesh_s": t_mesh, "solve_s": t_sol}


def _micro_mom_loc(G, conds, eps_r=11.45, ppg=3.0, *args, **kw):
    """micro_mom_loc + a certificate (certified mode). Window records of forked workers come back through a file."""
    LOG.clear()
    fd, path = tempfile.mkstemp(prefix="fp_cert_", suffix=".jsonl")
    os.close(fd)
    _CERT_FILE[0] = path
    try:
        out = _ORIG["micro_mom_loc"](G, conds, eps_r, ppg, *args, **kw)
        with open(path) as f:
            LOG.extend(json.loads(ln) for ln in f if ln.strip())
    finally:
        _CERT_FILE[0] = None
        os.unlink(path)
    CERTS.append(_certificate(G, conds, out[0], list(LOG), ppg))
    return out


def _certificate(G, conds, C, recs, ppg):
    """Calibrated estimate (c = RITZ_CAL) of the prior's solver error: max relative entry error (0.1 fF floor) over the
    accepted solves; per-entry fF estimates c sqrt(e_i e_j) for a single-window single-frame layout (input order).
    Self entries of Ritz solves are lower bounds of the exact ones."""
    acc = [r for r in recs if r.get("accepted", True) and "est_rel" in r]
    cert = dict(ppg=ppg, n_solves=len(recs), n_ritz=sum(bool(r.get("ritz")) for r in recs),
                n_exact=sum(bool(r.get("exact")) for r in recs), n_fallback=sum(bool(r.get("fallback")) for r in recs),
                max_rel_est=RITZ_CAL * max((r["est_rel"] for r in acc), default=0.0), calibration=RITZ_CAL,
                windows=[dict(window=r.get("window"), n=r["n"], ritz=bool(r.get("ritz")), max_rel_est=RITZ_CAL * r["est_rel"])
                         for r in acc], entry_est_fF=None)
    if len(acc) == 1 and len(conds) <= 8:
        frames = F.canonical_frames(G, conds)
        e = np.maximum(np.asarray(acc[0]["est_diag"], float), 0.0)
        if len(frames) == 1 and len(e) == len(conds) + 1:
            inv = np.r_[0, np.argsort(frames[0][2]) + 1]
            ei = e[inv]
            cert["entry_est_fF"] = (RITZ_CAL * np.sqrt(np.outer(ei, ei))).tolist()
    return cert


def certificates(clear=True) -> list:
    out = list(CERTS)
    if clear:
        CERTS.clear()
    return out


# ------------------------------------------------------------------------------------------------ install


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve(mode=None, components=None) -> tuple:
    if components is not None:
        bad = set(components) - set(COMPONENTS)
        if bad:
            raise ValueError(f"unknown components {sorted(bad)}; known {COMPONENTS}")
        return tuple(c for c in COMPONENTS if c in components)
    mode = mode or os.environ.get("TLSQ_PRIOR_MODE", "exact")
    if mode not in ("exact", "exact_compiled", "certified", "frozen"):
        raise ValueError(f"mode {mode!r}")
    if mode == "frozen":
        return ()
    comps = ["fv", "parallel", "hmat_vec"]
    if _gpu_available():
        comps.append("gpu")
    if MIXED_GATED:
        comps.append("mixed")
    if mode in ("exact_compiled", "certified"):
        comps.append("compiled")
    if mode == "certified":
        comps.append("ritz")
    return tuple(c for c in COMPONENTS if c in comps)


def install(mode=None, components=None, workers=None, cpus=None, target="auto", blas_threads="auto", value_patch=None,
            warm=True, **cfg):
    """Install the components (mode 'exact' / 'certified' / 'frozen', or an explicit component list) on the frozen
    vendor modules. workers: window workers (default TLSQ_PRIOR_WORKERS, else CPUs); cpus: default detect_cpus();
    blas_threads='auto': this process's BLAS / numba / pool threads = CPUs (None leaves them alone)."""
    uninstall()
    bind(target)
    CFG.update(cfg)
    comps = resolve(mode, components)
    STATE["mode"] = mode or ("explicit" if components is not None else os.environ.get("TLSQ_PRIOR_MODE", "exact"))
    STATE["components"] = comps
    STATE["cpus"] = int(cpus) if cpus else detect_cpus()
    w = workers if workers is not None else os.environ.get("TLSQ_PRIOR_WORKERS")
    # default (A5): serial windows with all threads; forked windows only for certified (measured faster there)
    STATE["workers"] = max(1, int(w)) if w is not None else (STATE["cpus"] if "ritz" in comps else 1)
    if "parallel" not in comps:
        STATE["workers"] = 1
    if sys.platform == "darwin" and STATE["workers"] > 1:  # Accelerate (BLAS) crashes in forked children: never fork on macOS
        STATE["workers"] = 1
        STATE["parallel_off"] = "darwin: the vendor's window pool forks and Accelerate is not fork-safe"
    STATE["gpu"] = "gpu" in comps
    _ORIG.update(window_solve=F._window_solve, loc_solve=FL._solve, micro_mom=F.micro_mom, lite_job=F._lite_job,
                 micro_mom_loc=FL.micro_mom_loc, mom_solve_fast=P.mom_solve_fast, assemble=P._assemble,
                 chol_solve=P._chol_solve, sym_solve=F._sym_solve, dense_max=P.DENSE_MAX, num_threads=P.NUM_THREADS)
    _ORIG["micro_mom_sig"] = inspect.signature(F.micro_mom)
    if blas_threads == "auto":
        _set_threads(STATE["cpus"])
    elif blas_threads:
        _set_threads(int(blas_threads))
    if "fv" in comps:
        F._window_solve = fv_window_solve
    F.micro_mom = _micro_mom
    F._lite_job = _lite_job
    _PARENT[0] = os.getpid()
    _CHILD[0] = False
    if "hmat_vec" in comps:
        _ORIG.update({"Kernel.__init__": P._Kernel.__init__, "Kernel.row": P._Kernel.row, "Kernel.block": P._Kernel.block, "aca": P._aca})
        P._Kernel.__init__, P._Kernel.row, P._Kernel.block, P._aca = _hv_kernel_init, _hv_row, _hv_block, _hv_aca
        P.mom_solve_fast = _hv_mom_solve_fast
    if "compiled" in comps:
        P.mom_solve_fast = _mom_solve_fast
        if warm:
            warmup()
    if "gpu" in comps:
        P._assemble = _gpu_assemble
        P.DENSE_MAX = CFG["gpu_dense_max"]
    if "gpu" in comps or "mixed" in comps:
        P._chol_solve = _chol_solve
        F._sym_solve = _sym_solve
    if "ritz" in comps:
        F._window_solve = ritz_window_solve
        FL._solve = ritz_loc_solve
        FL.micro_mom_loc = _micro_mom_loc
        if warm:
            warmup()
    if value_patch is not None:
        _UNDO.append(value_patch(F, P, FL))
        STATE["value_patch"] = getattr(value_patch, "__name__", str(value_patch))
    return dict(STATE)


def uninstall() -> None:
    for u in reversed(_UNDO):
        if callable(u):
            u()
    _UNDO.clear()
    if _ORIG:
        F._window_solve, FL._solve, F.micro_mom, F._lite_job = (_ORIG["window_solve"], _ORIG["loc_solve"], _ORIG["micro_mom"],
                                                               _ORIG["lite_job"])
        FL.micro_mom_loc, P.mom_solve_fast, P._assemble = _ORIG["micro_mom_loc"], _ORIG["mom_solve_fast"], _ORIG["assemble"]
        P._chol_solve, F._sym_solve, P.DENSE_MAX = _ORIG["chol_solve"], _ORIG["sym_solve"], _ORIG["dense_max"]
        P.NUM_THREADS = _ORIG["num_threads"]
        if "aca" in _ORIG:
            P._Kernel.__init__, P._Kernel.row, P._Kernel.block, P._aca = (_ORIG["Kernel.__init__"], _ORIG["Kernel.row"],
                                                                          _ORIG["Kernel.block"], _ORIG["aca"])
        _ORIG.clear()
    for tp in _TP:
        try:
            tp.restore_original_limits()
        except Exception:  # noqa: BLE001
            pass
    _TP.clear()
    STATE.update(mode=None, components=(), workers=1, gpu=False, value_patch=None, parallel_off=None)


@contextlib.contextmanager
def installed(**kw):
    install(**kw)
    try:
        yield dict(STATE)
    finally:
        uninstall()


def warmup() -> float:
    """Compile / load the numba kernels (first call per process; cached on disk afterwards). Returns seconds."""
    t = time.perf_counter()
    rng = np.random.default_rng(0)
    c = rng.random((600, 2)) * 100
    H = HFar(c, leaf=64)
    H(np.ones((600, 2)))
    Y = np.empty((600, 1))
    _numba()["far"](c[:, 0].copy(), c[:, 1].copy(), np.ones((1, 600)), Y)
    return time.perf_counter() - t


def status() -> dict:
    return dict(STATE, cfg=dict(CFG), mixed_gated=MIXED_GATED, frozen_dense_max=FROZEN_DENSE_MAX)
