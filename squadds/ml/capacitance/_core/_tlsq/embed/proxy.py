"""Cheap physics proxy: thin-film method-of-moments capacitance on the metal masks.

Planar conductors of negligible thickness on a dielectric half-space: the quasi-static Green's function is
1 / (4 pi eps0 eps_eff r) with eps_eff = (eps_r + 1) / 2. The metal is discretised into square cells (pulse basis,
point matching); a uniformly charged square of side a has self-potential q / (4 pi eps0 eps_eff) * 4 ln(1 + sqrt 2) / a.
Solving P q = v for each conductor at 1 V (others at 0 V) gives the full Maxwell capacitance matrix, including
shielding and other non-local effects that local descriptors cannot represent. Truncation: the ground plane is
represented only inside the design window (a proxy, not a replacement for Q3D).

v2 (`mom_capacitance_gds`) works on the exact GDS polygons with the whole ground plane and an adaptive quadtree mesh
(cells no larger than `grade` x their distance to a metal edge, edge cells down to ~1/3 of the smallest gap). On the
gallery designs it reaches 5-6 % (NCAP, star) and 12 % (concentric) median error with no training, vs 14-32 % for v1.
"""

from __future__ import annotations

import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import shapely
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

EPS0_FF_PER_UM = 8.8541878128e-3
SELF = 4 * np.log(1 + np.sqrt(2))
NUM_THREADS = 0  # threads of the mesh / assembly pool and of the concurrent v3-rich solves; 0 = torch.get_num_threads()
_LOCK = threading.Lock()
_POOL: list = []  # [(threads, executor)]
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_POOL.clear)  # a forked worker must not reuse the parent's pool threads


def _nthreads() -> int:
    if NUM_THREADS:
        return NUM_THREADS
    try:
        import torch
    except ImportError:
        return 1
    return torch.get_num_threads()


def _run(tasks: list) -> list:
    """Results of the zero-argument callables in order, on the thread pool when more than one thread is allowed (numpy
    ufuncs and shapely's vectorised functions release the GIL). Tasks never wait on other pool tasks."""
    n = _nthreads()
    if n < 2 or len(tasks) < 2:
        return [t() for t in tasks]
    with _LOCK:
        if not _POOL or _POOL[0][0] != n:
            _POOL[:] = [(n, ThreadPoolExecutor(n, thread_name_prefix="tlsq-mom"))]
        pool = _POOL[0][1]
    return list(pool.map(lambda t: t(), tasks))


def _spans(n: int, size: int) -> list:
    return [(a, min(a + size, n)) for a in range(0, n, size)]


class _Geoms:
    """Private copies of geometries (exact WKB round trip) handed out to one pool task at a time and made on demand: GEOS
    prepared geometries keep mutable search state and must not be shared between threads (nor with a concurrent solve)."""

    def __init__(self, geoms: list):
        self.wkb, self.free = shapely.to_wkb(geoms), queue.SimpleQueue()

    def get(self) -> list:
        try:
            return self.free.get_nowait()
        except queue.Empty:
            return list(shapely.from_wkb(self.wkb))

    def put(self, gs: list) -> None:
        self.free.put(gs)


def _cover(gq, boxes, h, tst, inn):
    """Coverage area(box & metal) / h^2 of every box by every metal (m, n), evaluated exactly only where a box straddles a
    metal: a box properly inside a metal covers 1 and a box disjoint from it 0 (same decisions as the exact areas). tst
    (m, n): metals the box can touch (its parent did); inn (m, n): metals the box lies properly inside (from its parent).
    Returns cov, hit (box intersects the metal), inn, and per metal the pieces box & metal where they were computed."""
    m, n = tst.shape
    cov, hit, inn = inn.astype(float), inn.copy(), inn.copy()
    pcs = [np.empty(n, dtype=object) for _ in range(m)]
    work = [(k, ix[a:b]) for k in range(m) for ix in [np.nonzero(tst[k])[0]] for a, b in _spans(len(ix), 256)]

    def task(k, ix):
        gs = gq.get()
        try:
            g = gs[k]
            shapely.prepare(g)
            b = boxes[ix]
            ht = shapely.intersects(g, b)
            it = shapely.contains_properly(g, b)
            p = shapely.intersection(b[ht & ~it], g)
            return ht, it, p, shapely.area(p)
        finally:
            gq.put(gs)

    for (k, ix), (ht, it, p, a) in zip(work, _run([partial(task, k, ix) for k, ix in work])):
        hit[k, ix], inn[k, ix] = ht, it
        cov[k, ix[it]] = 1.0
        s = ix[ht & ~it]
        cov[k, s] = a / h[s] ** 2
        pcs[k][s] = p
    return cov, hit, inn, pcs


def _distance(gq, j: int, geoms) -> np.ndarray:
    """shapely.distance(geoms, geometry j of the queue), in chunks on the pool."""
    def task(a, b):
        gs = gq.get()
        try:
            return shapely.distance(geoms[a:b], gs[j])
        finally:
            gq.put(gs)
    return np.concatenate([np.zeros(0)] + _run([partial(task, a, b) for a, b in _spans(len(geoms), 256)]))




def mom_capacitance(masks: np.ndarray, window_um: float, eps_r: float = 11.45, cells: int = 64) -> np.ndarray:
    """masks: (n_cond, R, R) coverage in [0, 1] (channel 0 = ground). Returns the (n_cond, n_cond) Maxwell C in fF."""
    nc, R, _ = masks.shape
    f = R // cells
    cov = masks.reshape(nc, cells, f, cells, f).mean((2, 4))  # coverage per coarse cell
    owner = np.where(cov.max(0) >= 0.5, cov.argmax(0), -1)
    ii, jj = np.nonzero(owner >= 0)
    lab = owner[ii, jj]
    a = window_um / cells
    xy = np.stack([(jj + 0.5) * a, (ii + 0.5) * a], 1)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 1.0)
    P = 1.0 / d
    np.fill_diagonal(P, SELF / a)
    P /= 4 * np.pi * EPS0_FF_PER_UM * (eps_r + 1) / 2
    B = (lab[:, None] == np.arange(nc)[None, :]).astype(float)  # conductor b at 1 V
    Q = cho_solve(cho_factor(P, lower=True), B)
    C = np.stack([Q[lab == a_].sum(0) for a_ in range(nc)])  # C[a, b] = charge on a with b at 1 V
    return 0.5 * (C + C.T)


def quadtree_cells(metals: list, bbox, h0: float, h_min: float, grade: float = 2.0, limit: int | None = None):
    """Square cells covering the metal polygons: a cell is kept whole when it lies inside one conductor and is no larger
    than `grade` x its distance to the nearest metal edge; edge cells are split down to h_min, where they go to the
    conductor covering >= 50 % of them. Returns centres (n, 2), sides (n,), owner (n,); None once more than `limit` cells."""
    edges = shapely.union_all([g.boundary for g in metals])
    x0, y0, x1, y1 = bbox
    nx, ny = int(np.ceil((x1 - x0) / h0)), int(np.ceil((y1 - y0) / h0))
    gx, gy = np.meshgrid(x0 + h0 * np.arange(nx), y0 + h0 * np.arange(ny))
    xs, ys, h = gx.ravel(), gy.ravel(), np.full(gx.size, h0)
    gq = _Geoms(list(metals) + [edges])
    tst, inn = np.ones((len(metals), len(xs)), bool), np.zeros((len(metals), len(xs)), bool)
    out_c, out_h, out_o, count = [], [], [], 0
    while len(xs):
        boxes = shapely.box(xs, ys, xs + h, ys + h)
        cov, hit, inn, _ = _cover(gq, boxes, h, tst, inn)
        keep = cov.sum(0) > 1e-3
        xs, ys, h, cov, hit, inn = xs[keep], ys[keep], h[keep], cov[:, keep], hit[:, keep], inn[:, keep]
        cx, cy = xs + h / 2, ys + h / 2
        full = cov.max(0) > 0.999
        d = np.full(len(xs), np.inf)  # the edge distance only decides for cells inside one conductor
        d[full] = _distance(gq, len(metals), shapely.points(cx[full], cy[full]))
        leaf = full & (h <= grade * d + 1e-12)
        last = h <= h_min * 1.0001
        take_last = last & ~leaf & (cov.max(0) >= 0.5)
        acc = leaf | take_last
        out_c.append(np.stack([cx[acc], cy[acc]], 1))
        out_h.append(h[acc])
        out_o.append(cov[:, acc].argmax(0))
        count += int(acc.sum())
        if limit is not None and count > limit:
            return None
        split = ~acc & ~last
        hs = h[split] / 2
        xs = np.concatenate([xs[split], xs[split] + hs, xs[split], xs[split] + hs])
        ys = np.concatenate([ys[split], ys[split], ys[split] + hs, ys[split] + hs])
        h = np.concatenate([hs] * 4)
        tst, inn = np.tile((hit & ~inn)[:, split], 4), np.tile(inn[:, split], 4)
    return np.concatenate(out_c), np.concatenate(out_h), np.concatenate(out_o)


def anchored_cells(metals: list, h_f: float, grade: float = 2.0, majority: bool = False, limit: int | None = None):
    """v3 mesh: quadtree on a lattice anchored to the signal conductors, independent of the chip size and position.

    Finest cells have side h_f; coarse cells are h_f * 2^k on the same lattice (origin = lower-left corner of the
    signal conductors' bounding box), so the cells near the conductors do not change when the ground plane grows,
    shrinks or the layout is translated. A cell is kept whole when it lies inside one conductor and is no larger than
    `grade` x the distance from the cell to the nearest metal edge (cells touching an edge reach h_f); finest-level cells that straddle an edge are cut along the exact
    polygon boundaries into one panel per metal (no rounding of gaps to the grid).
    Returns centres (n, 2), sizes (n, 2) = (width, height) of each panel, owner (n,); None once the mesh has more
    than `limit` cells."""
    edges = shapely.union_all([g.boundary for g in metals])
    sig = metals[1:] if len(metals) > 1 else metals
    ax, ay = shapely.union_all(sig).bounds[:2]
    X0, Y0, X1, Y1 = shapely.union_all(metals).bounds
    span = max(X1 - X0, Y1 - Y0)
    K = max(0, int(np.ceil(np.log2(max(span / 16, h_f) / h_f))))
    h0 = h_f * 2**K
    ix0, iy0 = int(np.floor((X0 - ax) / h0)), int(np.floor((Y0 - ay) / h0))
    ix1, iy1 = int(np.ceil((X1 - ax) / h0)), int(np.ceil((Y1 - ay) / h0))
    gx, gy = np.meshgrid(ax + h0 * np.arange(ix0, ix1), ay + h0 * np.arange(iy0, iy1))
    xs, ys, h = gx.ravel(), gy.ravel(), np.full(gx.size, h0)
    gq = _Geoms(list(metals) + [edges])
    tst, inn = np.ones((len(metals), len(xs)), bool), np.zeros((len(metals), len(xs)), bool)
    out_c, out_h, out_o, count = [], [], [], 0
    while len(xs):
        boxes = shapely.box(xs, ys, xs + h, ys + h)
        cov, hit, inn, pcs = _cover(gq, boxes, h, tst, inn)
        keep = cov.sum(0) > 1e-3
        xs, ys, h, cov, boxes, hit, inn = xs[keep], ys[keep], h[keep], cov[:, keep], boxes[keep], hit[:, keep], inn[:, keep]
        pcs = [p[keep] for p in pcs]
        cx, cy = xs + h / 2, ys + h / 2
        full = cov.max(0) > 0.999
        last = h <= h_f * 1.0001
        d = np.full(len(xs), np.inf)  # the edge distance only decides for whole cells above the finest level
        far = full & ~last
        d[far] = _distance(gq, len(metals), boxes[far])
        leaf = full & (last | (h <= grade * d + 1e-12))
        out_c.append(np.stack([cx[leaf], cy[leaf]], 1))
        out_h.append(np.stack([h[leaf], h[leaf]], 1))
        out_o.append(cov[:, leaf].argmax(0))
        count += int(leaf.sum())
        cut = last & ~leaf
        if majority and cut.any():
            take = cut & (cov.max(0) >= 0.5)
            out_c.append(np.stack([cx[take], cy[take]], 1))
            out_h.append(np.stack([h[take], h[take]], 1))
            out_o.append(cov[:, take].argmax(0))
            count += int(take.sum())
            cut = cut & False
        if cut.any():
            gs = gq.get()
            for k, g in enumerate(gs[:-1]):
                shapely.prepare(g)
                sel = np.nonzero(cut & (cov[k] > 1e-3))[0]
                if not len(sel):
                    continue
                pcs_k = pcs[k][sel]  # box & metal, computed by _cover (straddling cells only reach this level uncut)
                bb = shapely.bounds(pcs_k)
                cen = shapely.get_coordinates(shapely.centroid(pcs_k))
                inside = shapely.contains_xy(g, cen[:, 0], cen[:, 1])
                if not inside.all():
                    rp = shapely.get_coordinates(shapely.point_on_surface(pcs_k[~inside]))
                    cen[~inside] = rp
                out_c.append(cen)
                out_h.append(np.maximum(np.stack([bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]], 1), 1e-6))
                out_o.append(np.full(len(sel), k))
                count += len(sel)
            gq.put(gs)
        if limit is not None and count > limit:
            return None
        split = ~leaf & ~last
        hs = h[split] / 2
        xs = np.concatenate([xs[split], xs[split] + hs, xs[split], xs[split] + hs])
        ys = np.concatenate([ys[split], ys[split], ys[split] + hs, ys[split] + hs])
        h = np.concatenate([hs] * 4)
        tst, inn = np.tile((hit & ~inn)[:, split], 4), np.tile(inn[:, split], 4)
    return np.concatenate(out_c), np.concatenate(out_h), np.concatenate(out_o)


def mom_mesh(metals: list, max_cells: int = 8000, h_min: float | None = None, grade: float = 2.0, cells_per_gap: float = 3.0,
             mesh: str = "v2", offset: tuple = (0.0, 0.0)):
    """Mesh for the thin-film MoM. mesh="v3": conductor-anchored lattice with exact edge panels
    (`anchored_cells`); mesh="v2" (default, published priors): the original chip-anchored quadtree (reproduces the published v2 priors).
    Returns (centres, sides, owner, h_min)."""
    bbox = shapely.union_all(metals).bounds
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    if h_min is None:
        gap = min(shapely.distance(metals[i], metals[j]) for i in range(len(metals)) for j in range(i + 1, len(metals)))
        h_min = max(gap / cells_per_gap, size / 4096)
    if mesh == "v2" and any(offset):
        h0 = size / 16
        bbox = (bbox[0] - offset[0] * h0, bbox[1] - offset[1] * h0, bbox[2] + h0, bbox[3] + h0)
    while True:
        if mesh == "v3":
            r = anchored_cells(metals, h_min, grade, limit=max_cells)
        else:
            r = quadtree_cells(metals, bbox, size / 16, h_min, grade, limit=max_cells)
        if r is not None:
            return (*r, h_min)
        h_min *= 1.5 if mesh == "v2" else 2.0  # v3: stay on the same lattice


def rect_self(a, b):
    """Potential at the centre of a uniformly charged a x b rectangle per unit charge, times 4 pi eps (= SELF / a for a
    square)."""
    r = np.sqrt(a**2 + b**2)
    return (2 * a * np.log((b + r) / a) + 2 * b * np.log((a + r) / b)) / (a * b)


def _mom_solve_torch(c, h, o, n_metals, eps_r, dev):
    """Same assembly and Cholesky solve as `mom_solve`, in float64 torch on `dev` (opt-in via TLSQ_MOM_DEVICE)."""
    import torch

    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=dev)  # noqa: E731
    wh = np.stack([h, h], 1) if h.ndim == 1 else h
    C, WH = t(c), t(wh)
    hm = WH.max(1).values
    d = torch.cdist(C, C)
    d.fill_diagonal_(1.0)
    P = 1.0 / d
    near = d < 1.5 * torch.maximum(hm[:, None], hm[None, :])
    near.fill_diagonal_(False)
    del d
    ii, jj = torch.nonzero(near, as_tuple=True)
    del near
    if len(ii):
        off = (torch.arange(3, dtype=torch.float64, device=dev) - 1) / 3
        ox, oy = torch.meshgrid(off, off, indexing="xy")
        sx = C[jj, 0, None] + ox.reshape(-1)[None] * WH[jj, 0, None]
        sy = C[jj, 1, None] + oy.reshape(-1)[None] * WH[jj, 1, None]
        r = torch.sqrt((C[ii, 0, None] - sx) ** 2 + (C[ii, 1, None] - sy) ** 2)
        P[ii, jj] = (1.0 / torch.clamp(r, min=1e-9)).mean(1)
        P = 0.5 * (P + P.T)
    diag = t(rect_self(wh[:, 0], wh[:, 1]) if h.ndim == 2 else SELF / h)
    P.diagonal().copy_(diag)
    P /= 4 * np.pi * EPS0_FF_PER_UM * (eps_r + 1) / 2
    B = t((o[:, None] == np.arange(n_metals)[None, :]).astype(float))
    L, info = torch.linalg.cholesky_ex(P)
    del P
    if int(info):
        raise np.linalg.LinAlgError(f"{int(info)}-th leading minor not positive definite")
    return torch.cholesky_solve(B, L).cpu().numpy()


def _mom_solve_gpu_slot(c, h, o, n_metals, eps_r):
    """GPU solve if one of TLSQ_MOM_GPU_SLOTS slots (file locks shared by the worker processes) is free and the solve
    fits; None means: solve on the CPU instead."""
    import fcntl
    import os
    import tempfile

    import torch

    slots = int(os.environ.get("TLSQ_MOM_GPU_SLOTS", "3"))
    tag = os.environ.get("SLURM_JOB_ID", "local")
    for k in range(slots):
        f = open(os.path.join(tempfile.gettempdir(), f"tlsq_gpu_{tag}_{k}.lock"), "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            continue
        try:
            return _mom_solve_torch(c, h, o, n_metals, eps_r, os.environ["TLSQ_MOM_DEVICE"])
        except RuntimeError as e:  # includes torch.OutOfMemoryError
            if "out of memory" not in str(e).lower():
                raise
            return None
        finally:
            torch.cuda.empty_cache()
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
    return None


def _G(x, y):
    """Antiderivative of 1 / sqrt(x^2 + y^2) over x and y: x asinh(y / |x|) + y asinh(x / |y|) (0 on the axes)."""
    ax, ay = np.abs(x), np.abs(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = np.where(ax > 0, x * np.arcsinh(y / np.where(ax > 0, ax, 1.0)), 0.0)
        g2 = np.where(ay > 0, y * np.arcsinh(x / np.where(ay > 0, ay, 1.0)), 0.0)
    return g1 + g2


def rect_potential(p, cq, wh):
    """Exact potential (x 4 pi eps) at points p (n, 2) of uniformly charged rectangles (unit total charge) centred at cq
    with sizes wh (n, 2): (1 / (w h)) * integral of 1 / r over the rectangle. Equals rect_self at the centre."""
    x1, x2 = cq[:, 0] - wh[:, 0] / 2 - p[:, 0], cq[:, 0] + wh[:, 0] / 2 - p[:, 0]
    y1, y2 = cq[:, 1] - wh[:, 1] / 2 - p[:, 1], cq[:, 1] + wh[:, 1] / 2 - p[:, 1]
    return (_G(x2, y2) - _G(x1, y2) - _G(x2, y1) + _G(x1, y1)) / (wh[:, 0] * wh[:, 1])


def _mom_solve_exact_near(c, wh, o, n_metals, eps_r):
    """v3 panels: exact rectangle potential for near pairs (d < 1.5 max side) instead of the 3x3 point average. Used only
    when the 3x3 system is not positive definite (sliver edge panels on long fine combs), so priors that solved before
    are unchanged."""
    P = _assemble(c, wh, rect_self(wh[:, 0], wh[:, 1]), eps_r, rect_potential)
    B = (o[:, None] == np.arange(n_metals)[None, :]).astype(float)
    return _chol_solve(P, B)


def mom_solve(c, h, o, n_metals: int, eps_r: float = 11.45):
    """mom_solve_quad; for v3 panels (h of shape (n, 2)) a non-positive-definite system is re-solved with the exact
    near field (`_mom_solve_exact_near`)."""
    try:
        return mom_solve_quad(c, h, o, n_metals, eps_r)
    except np.linalg.LinAlgError:
        if h.ndim != 2:
            raise
        print("v3: 3x3 near-field system not positive definite, exact near field", flush=True)
        return _mom_solve_exact_near(c, h, o, n_metals, eps_r)


def mom_solve_quad(c, h, o, n_metals: int, eps_r: float = 11.45):
    """Point-matching solve on a mesh from `mom_mesh`: cell charges Q (n_cells, n_metals) for each conductor at 1 V.
    h: square sides (n,) or panel (width, height) (n, 2). Set TLSQ_MOM_DEVICE=cuda to assemble and solve on a GPU."""
    import os

    if os.environ.get("TLSQ_MOM_DEVICE") and SUBSTRATE["model"] == "halfspace":  # the torch path has no image correction
        Q = _mom_solve_gpu_slot(c, h, o, n_metals, eps_r)
        if Q is not None:
            return Q
    wh = np.stack([h, h], 1) if h.ndim == 1 else h
    P = _assemble(c, wh, rect_self(wh[:, 0], wh[:, 1]) if h.ndim == 2 else SELF / h, eps_r, _near_quad)
    B = (o[:, None] == np.arange(n_metals)[None, :]).astype(float)
    return _chol_solve(P, B)


def _near_quad(p, cq, wh):
    """Potential (x 4 pi eps) at points p of the panels centred at cq with sizes wh: mean of 1 / r over a 3x3 sample of the
    panel (point matching is poor for close, unequal cells)."""
    off = (np.arange(3) - 1) / 3
    ox, oy = np.meshgrid(off, off)
    sx = cq[:, 0, None] + ox.ravel()[None] * wh[:, 0, None]
    sy = cq[:, 1, None] + oy.ravel()[None] * wh[:, 1, None]
    r = np.sqrt((p[:, 0, None] - sx) ** 2 + (p[:, 1, None] - sy) ** 2)
    return (1.0 / np.maximum(r, 1e-9)).mean(1)


def _near_pairs(c, hm):
    """Pairs i < j with |c_i - c_j| < 1.5 max(hm_i, hm_j): candidates from k-d trees per cell-size class, kept by the same
    floating-point test as the dense mask."""
    n = len(c)
    tree, lev, key = cKDTree(c), np.floor(np.log2(hm)), [np.zeros(0, np.int64)]
    for L in np.unique(lev):
        sel = np.nonzero(lev == L)[0]
        pr = cKDTree(c[sel]).sparse_distance_matrix(tree, 1.5 * hm[sel].max() * (1 + 1e-9), output_type="ndarray")
        a, b = sel[pr["i"]], pr["j"]
        key.append(np.minimum(a, b) * n + np.maximum(a, b))
    key = np.unique(np.concatenate(key))
    ii, jj = key // n, key % n
    ok = (ii != jj) & (np.sqrt((c[ii, 0] - c[jj, 0]) ** 2 + (c[ii, 1] - c[jj, 1]) ** 2) < 1.5 * np.maximum(hm[ii], hm[jj]))
    return ii[ok], jj[ok]


SUBSTRATE = {"model": "slab_open", "t_um": 350.0}  # MoM Green's function (PRIOR_SCHEME.md 14.6): "slab_open" (default since 2026-10-10: 350 um substrate, vacuum below), "halfspace" (released models' priors), "slab_pec"
_IMG_CACHE: OrderedDict = OrderedDict()  # LRU of image tables (model, t_um, eps_r) -> (du, T, w, D)
_IMG_GRID: dict = {}
_IMG_KEEP = (4, 512 << 20)  # at most 4 tables and 512 MB (the newest is always kept)
_INTERP: dict = {}


def _img_evict():
    while len(_IMG_CACHE) > 1 and (len(_IMG_CACHE) > _IMG_KEEP[0] or sum(v[1].nbytes for v in _IMG_CACHE.values())
                                   + sum(g.nbytes for g in _IMG_GRID.values()) > _IMG_KEEP[1]):
        _IMG_GRID.pop(_IMG_CACHE.popitem(last=False)[0], None)


try:  # compiled np.interp on the uniform image table (_interp_uniform)
    import numba as _nb
    from numba.core import types as _nbt
    from numba.extending import intrinsic as _intrinsic

    @_intrinsic
    def _fma(typingctx, a, b, c):
        def gen(context, builder, sig, args):
            return builder.fma(*args)
        return _nbt.float64(_nbt.float64, _nbt.float64, _nbt.float64), gen

    @_nb.njit(nogil=True, cache=True)
    def _interp_run(x, du, fp, out, fused):
        """numpy's arr_interp on xp_j = du * j: the index by division, corrected with numpy's comparisons, then numpy's slope
        and slope * (x - xp_j) + fp_j, plain or fused (fma) as this numpy build's C compiler does it."""
        n = fp.shape[0]
        last = du * (n - 1)
        for i in range(x.shape[0]):
            v = x[i]
            if v != v:
                out[i] = v
                continue
            if v > last:
                out[i] = fp[n - 1]
                continue
            if v < 0.0:
                out[i] = fp[0]
                continue
            j = min(int(v / du), n - 1)
            while j > 0 and du * j > v:
                j -= 1
            while j < n - 1 and du * (j + 1) <= v:
                j += 1
            xj = du * j
            if j == n - 1 or xj == v:
                out[i] = fp[j]
                continue
            x1 = du * (j + 1)
            s = (fp[j + 1] - fp[j]) / (x1 - xj)
            r = _fma(s, v - xj, fp[j]) if fused else s * (v - xj) + fp[j]
            if r != r:
                r = _fma(s, v - x1, fp[j + 1]) if fused else s * (v - x1) + fp[j + 1]
                if r != r and fp[j] == fp[j + 1]:
                    r = fp[j]
            out[i] = r
except ImportError:
    _interp_run = None


def _interp_uniform(d, du, T):
    """np.interp(d, du * arange(len(T)), T), bit for bit: the compiled kernel (plain or fused multiply-add, whichever matches this
    numpy build) when a probe at first use matches np.interp bit for bit, else None (the caller uses np.interp)."""
    if "fn" not in _INTERP:
        _INTERP["fn"] = None
        if _interp_run is not None:
            try:
                rng, ok = np.random.default_rng(0), []
                for fp in (rng.normal(size=4096), T):  # a rough table tells the two apart, then the real one
                    xp = du * np.arange(len(fp))
                    x = np.concatenate([rng.uniform(-1, 1.01 * xp[-1], 50000), rng.uniform(0, 64 * du, 20000), xp[rng.integers(0, len(fp), 2000)],
                                        np.nextafter(xp[rng.integers(1, len(fp), 2000)], 0), xp[[0, -1]], [np.nan, np.inf, 1e300]])
                    ref, got = np.interp(x, xp, fp).tobytes(), []
                    for fused in (False, True):
                        y = np.empty_like(x)
                        _interp_run(x, du, fp, y, fused)
                        got.append(y.tobytes())
                    ok.append([g == ref for g in got])
                if ok[0][0] != ok[0][1] and ok[1][ok[0][1]]:
                    _INTERP["fn"] = ok[0][1]
            except Exception:  # noqa: BLE001  (llvm.fma or the compile unavailable: np.interp)
                pass
    fused = _INTERP["fn"]
    d = np.asarray(d, float)
    if fused is None or d.ndim == 0:
        return None
    x = np.ascontiguousarray(d)
    out = np.empty(x.shape)
    _interp_run(x.reshape(-1), du, T, out.reshape(-1), fused)
    return out


class substrate:
    """Context manager: the MoM Green's function inside the block (`SUBSTRATE`): model "slab_open" (default since
    2026-10-10, PRIOR_SCHEME.md 14.6), "halfspace" (the kernel of the priors the released models were trained on) or
    "slab_pec"; t_um the substrate thickness (None: unchanged)."""

    def __init__(self, model: str | None, t_um: float | None = None):
        self.new = {k: v for k, v in (("model", model), ("t_um", t_um)) if v is not None}

    def __enter__(self):
        self.old = dict(SUBSTRATE)
        SUBSTRATE.update(self.new)
        return self

    def __exit__(self, *exc):
        SUBSTRATE.clear()
        SUBSTRATE.update(self.old)


def _image_correction(d, eps_r: float, exact: bool = False):
    """Finite-substrate correction to the half-space kernel 1 / d (same units): sum_n w_n / sqrt(d^2 + D_n^2) for panels on the
    top face of a slab of thickness t (SUBSTRATE): "slab_open" = vacuum below (w_n = K^2n + K^(2n-1), D_n = 2 n t,
    K = (eps_r - 1) / (eps_r + 1); weights sum to eps_eff - 1, so the kernel tends to the vacuum one for d >> eps_r t),
    "slab_pec" = metal backside (w_n = -(1 + K) (-K)^(n-1); weights sum to -1). Images up to K^2n < 1e-4, the remaining
    weight as one more image; tabulated on a uniform grid of t / 1000 up to 100 mm and interpolated linearly (dense
    assembly), or summed exactly (exact=True: the H-matrix kernel, whose ACA blocks need a smooth kernel). None for the
    half-space (the published kernel)."""
    m = SUBSTRATE["model"]
    if m == "halfspace":
        return None
    key = (m, float(SUBSTRATE["t_um"]), float(eps_r))
    if key in _IMG_CACHE:
        _IMG_CACHE.move_to_end(key)
    else:
        K, t = (eps_r - 1) / (eps_r + 1), float(SUBSTRATE["t_um"])
        n = np.arange(1, 400)
        nmax = int(np.argmax(K ** (2 * n) < 1e-4)) + 1
        n = n[:nmax]
        if m == "slab_open":
            w, total = K ** (2 * n) + K ** (2 * n - 1), K / (1 - K)
        else:
            w, total = -(1 + K) * (-K) ** (n - 1), -1.0
        w, D = np.r_[w, total - w.sum()], np.r_[2 * n * t, 2 * (nmax + 1) * t]
        du = t / 1000
        u = np.arange(0.0, 1e5 + 2 * du, du)
        T = np.empty(len(u))
        for i in range(0, len(u), 1 << 18):  # row blocks: the same sums, without len(u) x images temporaries (~8 GB at 10 um)
            ub = u[i:i + (1 << 18)]
            T[i:i + len(ub)] = (w[None, :] / np.sqrt(ub[:, None] ** 2 + D[None, :] ** 2)).sum(1)
        _IMG_CACHE[key] = (du, T, w, D)
        _img_evict()
    du, T = _IMG_CACHE[key][:2]
    if exact:
        w, D = _IMG_CACHE[key][2:]
        d2 = np.asarray(d, float) ** 2
        return sum(wn / np.sqrt(d2 + Dn * Dn) for wn, Dn in zip(w, D))
    out = _interp_uniform(d, du, T)
    if out is not None:
        return out
    if key not in _IMG_GRID:
        _IMG_GRID[key] = du * np.arange(len(T))
    return np.interp(d, _IMG_GRID[key], T)


def _assemble(c, wh, diag, eps_r: float, near):
    """Interaction matrix / (4 pi eps0 eps_eff) for the lower Cholesky factorisation: far pairs 1 / d, near pairs (d < 1.5
    max side) the symmetrised near(p, cq, wh) potential, diagonal `diag`; the same numbers as the dense construction.
    Built in row blocks on the pool threads without n x n temporaries, and only the half the factorisation reads: returns
    a Fortran-ordered matrix whose lower triangle holds the entries (its upper triangle is not initialised)."""
    n = len(c)
    k = 4 * np.pi * EPS0_FF_PER_UM * (eps_r + 1) / 2
    A = np.empty((n, n))
    x, y = c[:, 0], c[:, 1]

    def rows(r0, r1):
        d = x[r0:r1, None] - x[None, r0:]
        np.multiply(d, d, out=d)
        t = y[r0:r1, None] - y[None, r0:]
        np.multiply(t, t, out=t)
        d += t
        np.sqrt(d, out=d)
        corr = _image_correction(d, eps_r)
        d[np.arange(r1 - r0), np.arange(r1 - r0)] = 1.0
        out = A[r0:r1, r0:]
        np.divide(1.0, d, out=out)
        if corr is not None:
            out += corr
        out /= k

    _run([partial(rows, a, b) for a, b in _spans(n, 64)])
    ii, jj = _near_pairs(c, wh.max(1))
    img = _image_correction(np.zeros(1), eps_r)
    if len(ii):
        A[ii, jj] = 0.5 * (near(c[ii], c[jj], wh[jj]) + near(c[jj], c[ii], wh[ii])) / k
        if img is not None:
            A[ii, jj] += _image_correction(np.sqrt(((c[ii] - c[jj]) ** 2).sum(1)), eps_r) / k
    A[np.arange(n), np.arange(n)] = diag / k if img is None else (diag + img[0]) / k
    return A.T


def _chol_solve(P, B):
    """Cholesky solve with the lower triangle of the Fortran-ordered P from `_assemble`, factorised in place (no copy)."""
    return cho_solve(cho_factor(P, lower=True, overwrite_a=True, check_finite=False), B, check_finite=False)


def mom_capacitance_gds(metals: list, eps_r: float = 11.45, max_cells: int = 8000, h_min: float | None = None, grade: float = 2.0,
                        cells_per_gap: float = 3.0, offset: tuple = (0.0, 0.0), mesh: str = "v2"):
    """Thin-film MoM on the exact GDS polygons (ground to the chip edge), adaptive quadtree mesh (see `mom_mesh`).

    metals: shapely geometries, one per conductor (index 0 = ground). `offset` (v2 mesh only) shifts the quadtree origin
    by that fraction of the coarsest cell. Returns (C fF in the same order, n_cells, h_min)."""
    c, h, o, h_min = mom_mesh(metals, max_cells, h_min, grade, cells_per_gap, mesh, offset)
    try:
        Q = mom_solve(c, h, o, len(metals), eps_r)  # v3: exact near field if the 3x3 system is not positive definite
    except np.linalg.LinAlgError:
        if mesh != "v3":
            raise
        # cell cap coarser than the finger pitch: cut-panel boxes of different conductors overlap; finest cells as whole
        # squares owned by the metal covering >= 50 % (as the v2 mesh), exact near field
        print("v3: still not positive definite, majority finest cells", flush=True)
        c, h, o = anchored_cells(metals, h_min, grade, majority=True)
        Q = _mom_solve_exact_near(c, h, o, len(metals), eps_r)
    C = np.stack([Q[o == a].sum(0) for a in range(len(metals))])
    return 0.5 * (C + C.T), len(c), h_min


def mom_capacitance_gds_avg(metals: list, n_offsets: int = 4, **kw):
    """MoM averaged over quadtree origins (offsets on a diagonal grid of the coarsest cell): removes the part of the
    discretisation error that jumps with mesh alignment as the geometry is swept."""
    offs = [(i / n_offsets, ((i * 3) % n_offsets) / n_offsets) for i in range(n_offsets)]
    runs = [mom_capacitance_gds(metals, offset=o, mesh="v2", **kw) for o in offs]
    return np.mean([r[0] for r in runs], 0), int(np.mean([r[1] for r in runs])), runs[0][2]


def _both(f, g):
    """(f(), g()), a LinAlgError in place of a result; concurrently (g on a helper thread) when more than one thread is
    allowed: the gap/6 and gap/3 solves of v3-rich are independent."""
    def call(fn):
        try:
            return fn()
        except np.linalg.LinAlgError as e:
            return e
    if _nthreads() < 2:
        return call(f), call(g)
    with ThreadPoolExecutor(1, thread_name_prefix="tlsq-rich") as ex:
        fut = ex.submit(call, g)
        a = call(f)
        return a, fut.result()


def mom_capacitance_gds_rich(metals: list, eps_r: float = 11.45, cells_per_gap: float = 3.0, max_cells: int = 8000,
                             max_cells_fine: int = 16000, grade: float = 2.0):
    """v3 MoM with Richardson extrapolation: solves at gap/cells_per_gap and at half that cell size on the same
    conductor-anchored lattice and extrapolates the first-order discretisation error to zero cell size. If the cell
    cap forces both meshes to the same resolution, returns the finer solve. If the coarse system is not positive
    definite (sliver edge panels badly resolved by the 3x3 near-field rule; ~0.2 % of designs), returns the other
    solve alone (fine if the coarse one fails, coarse if the fine one fails). The two solves run concurrently when more
    than one thread is allowed. Returns (C, total cells, h of the fine mesh)."""
    rf, rc = _both(partial(mom_capacitance_gds, metals, eps_r, max_cells_fine, None, grade, 2 * cells_per_gap, mesh="v3"),
                   partial(mom_capacitance_gds, metals, eps_r, max_cells, None, grade, cells_per_gap, mesh="v3"))
    if isinstance(rf, np.linalg.LinAlgError):
        print("v3-rich: fine mesh not positive definite, coarse solve only", flush=True)
        if isinstance(rc, np.linalg.LinAlgError):
            raise rc
        return rc
    Cf, nf, hf = rf
    if isinstance(rc, np.linalg.LinAlgError):
        print("v3-rich: coarse mesh not positive definite, fine solve only", flush=True)
        return Cf, nf, hf
    Cc, nc, hc = rc
    r = hc / hf
    C = Cf if r < 1.01 else Cf + (Cf - Cc) / (r - 1)
    return C, nc + nf, hf


# ---------------------------------------------------------------- flagship prior: strip-merged mesh, H-matrix solve, Richardson
# Resolution-controlled (panels per gap, no small cap), the same conductor-anchored panels as v3 merged along straight
# edges, the exact rectangle potential for near pairs, and for large systems an H-matrix (ACA far field) with
# preconditioned CG instead of the dense Cholesky: O(n log n) time and memory.

DENSE_MAX = int(os.environ.get("TLSQ_MOM_DENSE_MAX", 12000))  # panels up to which the dense Cholesky is used (faster below this size)
HMAT = dict(leaf=128, eta=2.0, eps=1e-9, tol=1e-12, maxit=1000)  # ACA block tolerance (relative Frobenius) and PCG residual


def _segments(metals):
    segs, verts = [], []
    for g in metals:
        for p in getattr(g, "geoms", [g]):
            for ring in [p.exterior, *p.interiors]:
                xy = np.asarray(ring.coords)
                verts.append(xy[:-1])
                segs.append(np.stack([xy[:-1], xy[1:]], 1))
    S = np.concatenate(segs)
    return shapely.linestrings(S), S, np.concatenate(verts)


def strip_merge(c, wh, o, metals, h_f, lmax_mult: float = 16.0, grade: float = 1.0):
    """Merge runs of equal-width panels along the direction of their nearest polygon edge (where the charge density is
    translation-invariant) into panels of length <= lmax_mult * h_f and <= grade * distance to the nearest vertex;
    the size across the edge is kept. Returns merged (centres, sizes, owners)."""
    lines, S, verts = _segments(metals)
    near = shapely.STRtree(lines).nearest(shapely.points(c))
    d = S[near, 1] - S[near, 0]
    horiz = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    dv, _ = cKDTree(verts).query(c, workers=-1)
    allow = np.minimum(lmax_mult * h_f, grade * dv)
    out_c, out_wh, out_o = [], [], []
    for axis, mask in ((0, horiz), (1, ~horiz)):
        if not mask.any():
            continue
        idx = np.nonzero(mask)[0]
        other = 1 - axis
        key = np.stack([o[idx], np.round(c[idx, other], 4), np.round(wh[idx, other], 4)], 1)
        order = np.lexsort([c[idx, axis], key[:, 2], key[:, 1], key[:, 0]])
        idx, key = idx[order], key[order]
        same = np.all(key[1:] == key[:-1], 1)
        lo, hi = c[idx, axis] - wh[idx, axis] / 2, c[idx, axis] + wh[idx, axis] / 2
        adjacent = same & (np.abs(hi[:-1] - lo[1:]) < 1e-6)
        start = 0
        while start < len(idx):
            end, length, cap = start, wh[idx[start], axis], allow[idx[start]]
            while end + 1 < len(idx) and adjacent[end] and length + wh[idx[end + 1], axis] <= min(cap, allow[idx[end + 1]]):
                end += 1
                length += wh[idx[end], axis]
                cap = min(cap, allow[idx[end]])
            cc, ww = np.zeros(2), np.zeros(2)
            cc[axis], cc[other] = 0.5 * (lo[start] + hi[end]), c[idx[start], other]
            ww[axis], ww[other] = hi[end] - lo[start], wh[idx[start], other]
            out_c.append(cc)
            out_wh.append(ww)
            out_o.append(o[idx[start]])
            start = end + 1
    return np.array(out_c), np.array(out_wh), np.array(out_o)


def mesh_flagship(metals: list, h: float, max_cells: int, grade: float = 2.0, lmax_mult: float = 16.0, merge_grade: float = 1.0,
                  majority: bool = False):
    """Strip-merged v3 mesh at finest panel h, doubled (same lattice) until the merged mesh has at most max_cells panels.
    Returns (centres, sizes, owners, h, raw panel count)."""
    while True:
        raw = anchored_cells(metals, h, grade, majority=majority, limit=8 * max_cells)
        if raw is not None:
            c, wh, o = strip_merge(*raw, metals, h, lmax_mult, merge_grade)
            if len(c) <= max_cells:
                return c, wh, o, h, len(raw[0])
        h *= 2.0


class _Kernel:
    """Entries of P / (4 pi eps0 eps_eff) in a permuted panel order: 1 / d for far pairs, the exact rectangle potential
    for near pairs (d < 1.5 max side) and rect_self on the diagonal; near and diagonal corrections to 1 / d are held in
    a CSR matrix so that any block is 1 / d + N[I, J] (the same numbers as `_assemble` with `rect_potential`)."""

    def __init__(self, c, wh, eps_r, perm):
        from scipy.sparse import csr_matrix

        self.x, self.y = c[perm, 0].copy(), c[perm, 1].copy()
        self.k = 4 * np.pi * EPS0_FF_PER_UM * (eps_r + 1) / 2
        self.eps_r = eps_r
        self.img = _image_correction(np.zeros(1), eps_r) is not None  # finite-substrate image term (exact series in blocks / rows)
        n = len(c)
        ii, jj = _near_pairs(c, wh.max(1))
        d = np.sqrt(((c[ii] - c[jj]) ** 2).sum(1))
        near = 0.5 * (rect_potential(c[ii], c[jj], wh[jj]) + rect_potential(c[jj], c[ii], wh[ii])) - 1.0 / d
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        rows = inv[np.concatenate([ii, jj, np.arange(n)])]
        cols = inv[np.concatenate([jj, ii, np.arange(n)])]
        vals = np.concatenate([near, near, rect_self(wh[:, 0], wh[:, 1]) - 1.0])
        self.N = csr_matrix((vals, (rows, cols)), shape=(n, n))

    def block(self, a0, a1, b0, b1):
        d = np.sqrt((self.x[a0:a1, None] - self.x[None, b0:b1]) ** 2 + (self.y[a0:a1, None] - self.y[None, b0:b1]) ** 2)
        corr = _image_correction(d, self.eps_r, exact=True)
        if a0 == b0:
            d[np.arange(a1 - a0), np.arange(a1 - a0)] = 1.0
        A = 1.0 / d
        if corr is not None:
            A += corr
        Nb = self.N[a0:a1, b0:b1]
        if Nb.nnz:
            A += Nb.toarray()
        return A / self.k

    def row(self, i, b0, b1):
        d = np.sqrt((self.x[i] - self.x[b0:b1]) ** 2 + (self.y[i] - self.y[b0:b1]) ** 2)
        r = 1.0 / d
        if self.img:
            r += _image_correction(d, self.eps_r, exact=True)
        Nr = self.N[i, b0:b1]
        if Nr.nnz:
            r += Nr.toarray()[0]
        return r / self.k


def _cluster_tree(c, leaf: int):
    """Recursive bisection of the panel centres along the longer side of their bounding box at the median: a permutation
    making every cluster a contiguous range, and nodes (a, b, lo, hi, left, right)."""
    perm, nodes = np.arange(len(c)), []

    def rec(a, b):
        idx = perm[a:b]
        lo, hi = c[idx].min(0), c[idx].max(0)
        k = len(nodes)
        nodes.append([a, b, lo, hi, -1, -1])
        if b - a > leaf:
            ax = int(np.argmax(hi - lo))
            perm[a:b] = idx[np.argsort(c[idx, ax], kind="stable")]
            m = a + (b - a) // 2
            nodes[k][4], nodes[k][5] = rec(a, m), rec(m, b)
        return k

    rec(0, len(c))
    return perm, nodes


def _aca(K, a0, a1, b0, b1, eps, max_rank):
    """Partial-pivoting adaptive cross approximation of the block (a0:a1, b0:b1) to relative Frobenius accuracy eps:
    (U, V) with block ~ U V^T, or None when the rank would exceed max_rank."""
    m, n = a1 - a0, b1 - b0
    U, V, used_r, used_c = [], [], np.zeros(m, bool), np.zeros(n, bool)
    i, norm2 = 0, 0.0
    for _ in range(min(m, n, max_rank)):
        r = K.row(a0 + i, b0, b1)
        for u, v in zip(U, V):
            r -= u[i] * v
        rm = np.where(used_c, 0.0, np.abs(r))
        j = int(np.argmax(rm))
        if rm[j] == 0.0:
            break
        v = r / r[j]
        u = K.row(b0 + j, a0, a1)  # column j of the block = row j of the symmetric kernel
        for uu, vv in zip(U, V):
            u -= vv[j] * uu
        used_r[i], used_c[j] = True, True
        nu, nv = float(u @ u), float(v @ v)
        norm2 += nu * nv + 2 * sum(float(u @ uu) * float(v @ vv) for uu, vv in zip(U, V))
        U.append(u)
        V.append(v)
        if nu * nv <= eps**2 * norm2:
            break
        um = np.where(used_r, 0.0, np.abs(u))
        i = int(np.argmax(um))
        if um[i] == 0.0:
            break
    else:
        return None
    return np.stack(U, 1), np.stack(V, 1)


class _HMatrix:
    """Hierarchical matrix of the panel system: dense blocks for leaf pairs that are not well separated, ACA low-rank
    blocks for admissible pairs (min diameter <= eta x distance); only the upper block triangle is stored (symmetric)."""

    def __init__(self, c, wh, eps_r, leaf=128, eta=2.0, eps=1e-9):
        self.n = len(c)
        self.perm, nodes = _cluster_tree(c, leaf)
        self.K = _Kernel(c, wh, eps_r, self.perm)
        self.eps = eps
        plan = []

        def leafq(k):
            return nodes[k][4] < 0

        def rec(s, t):
            a0, a1, lo_s, hi_s, sl, sr = nodes[s]
            b0, b1, lo_t, hi_t, tl, tr = nodes[t]
            if s == t:
                if leafq(s):
                    plan.append(("d", a0, a1, a0, a1))
                else:
                    rec(sl, sl), rec(sr, sr), rec(sl, sr)
                return
            gap = np.maximum(0.0, np.maximum(lo_s - hi_t, lo_t - hi_s))
            diam = min(np.sqrt(((hi_s - lo_s) ** 2).sum()), np.sqrt(((hi_t - lo_t) ** 2).sum()))
            if diam <= eta * np.sqrt((gap**2).sum()):
                plan.append(("l", a0, a1, b0, b1))
            elif leafq(s) and leafq(t):
                plan.append(("d", a0, a1, b0, b1))
            elif leafq(s):
                rec(s, tl), rec(s, tr)
            elif leafq(t):
                rec(sl, t), rec(sr, t)
            else:
                for u in (sl, sr):
                    for v in (tl, tr):
                        rec(u, v)

        rec(0, 0)
        self.dense, self.low = [], []

        def build(chunk):
            out = []
            for kind, a0, a1, b0, b1 in chunk:
                if kind == "l":
                    r = _aca(self.K, a0, a1, b0, b1, eps, max(2, min(a1 - a0, b1 - b0) // 2))
                    if r is not None:
                        out.append(("l", a0, a1, b0, b1, r))
                        continue
                out.append(("d", a0, a1, b0, b1, self.K.block(a0, a1, b0, b1)))
            return out

        for res in _run([partial(build, plan[a:b]) for a, b in _spans(len(plan), 64)]):
            for kind, a0, a1, b0, b1, M in res:
                (self.low if kind == "l" else self.dense).append((a0, a1, b0, b1, M))
        self.pre = [(a0, a1, cho_factor(D, lower=True)) for a0, a1, b0, _, D in self.dense if a0 == b0]  # block Jacobi
        self.n_threads = max(1, _nthreads())


    def nbytes(self):
        return sum(D.nbytes for *_, D in self.dense) + sum(U.nbytes + V.nbytes for *_, (U, V) in self.low)

    def matvec(self, X):
        def part(dense, low):
            Y = np.zeros_like(X)
            for a0, a1, b0, b1, D in dense:
                Y[a0:a1] += D @ X[b0:b1]
                if a0 != b0:
                    Y[b0:b1] += D.T @ X[a0:a1]
            for a0, a1, b0, b1, (U, V) in low:
                Y[a0:a1] += U @ (V.T @ X[b0:b1])
                Y[b0:b1] += V @ (U.T @ X[a0:a1])
            return Y

        T = self.n_threads
        if T < 2:
            return part(self.dense, self.low)
        return sum(_run([partial(part, self.dense[k::T], self.low[k::T]) for k in range(T)]))

    def precondition(self, R):
        Z = np.empty_like(R)
        for a0, a1, cf in self.pre:
            Z[a0:a1] = cho_solve(cf, R[a0:a1], check_finite=False)
        return Z

    def pcg(self, B, tol, maxit):
        """Conjugate gradients on every right-hand side column at once (independent columns), block-Jacobi preconditioned,
        until every column's relative residual is below tol. B and the result are in the permuted order."""
        X, R = np.zeros_like(B), B.copy()
        Z = self.precondition(R)
        P, rz = Z.copy(), (R * Z).sum(0)
        bn = np.sqrt((B * B).sum(0))
        for it in range(maxit):
            AP = self.matvec(P)
            alpha = rz / (P * AP).sum(0)
            X += alpha * P
            R -= alpha * AP
            res = (np.sqrt((R * R).sum(0)) / bn).max()
            if res < tol:
                return X, it + 1, res
            Z = self.precondition(R)
            rz_new = (R * Z).sum(0)
            P = Z + (rz_new / rz) * P
            rz = rz_new
        raise np.linalg.LinAlgError(f"H-matrix PCG did not converge in {maxit} iterations (residual {res:.1e})")


def mom_solve_fast(c, wh, o, n_metals: int, eps_r: float = 11.45, dense_max: int | None = None):
    """Panel charges Q (n, n_metals) for unit excitation of every conductor, exact near field: dense Cholesky for
    n <= dense_max panels (`_mom_solve_exact_near`), otherwise the H-matrix / PCG solve of the same system."""
    n = len(c)
    if n <= (DENSE_MAX if dense_max is None else dense_max):
        return _mom_solve_exact_near(c, wh, o, n_metals, eps_r)
    H = _HMatrix(c, wh, eps_r, HMAT["leaf"], HMAT["eta"], HMAT["eps"])
    B = (o[:, None] == np.arange(n_metals)[None, :]).astype(float)[H.perm]
    Xp, _, _ = H.pcg(B, HMAT["tol"], HMAT["maxit"])
    Q = np.empty_like(Xp)
    Q[H.perm] = Xp
    return Q


def _cap_flagship(metals, eps_r, h, max_cells, grade, lmax_mult, merge_grade):
    c, wh, o, h, n_raw = mesh_flagship(metals, h, max_cells, grade, lmax_mult, merge_grade)
    try:
        Q = mom_solve_fast(c, wh, o, len(metals), eps_r)
    except np.linalg.LinAlgError:  # cut panels of different conductors overlap (lattice coarser than the finger pitch)
        print("flagship: system not positive definite, majority finest cells", flush=True)
        c, wh, o, h, n_raw = mesh_flagship(metals, h, max_cells, grade, lmax_mult, merge_grade, majority=True)
        Q = mom_solve_fast(c, wh, o, len(metals), eps_r)
    C = np.stack([Q[o == a].sum(0) for a in range(len(metals))])
    return 0.5 * (C + C.T), len(c), h, n_raw


def mom_capacitance_gds_flagship(metals: list, eps_r: float = 11.45, cells_per_gap: float = 3.0, max_cells: int = 250000,
                                 grade: float = 2.0, rich: bool = True, lmax_mult: float = 16.0, merge_grade: float = 1.0):
    """Flagship prior: strip-merged conductor-anchored mesh at h_f = max(gap / (2 cells_per_gap), size / 8192) and, with
    rich=True, Richardson extrapolation with the mesh at 2 h_f (same lattice), both solved with `mom_solve_fast`. The
    resolution is set by cells_per_gap (the fine mesh has 2 cells_per_gap panels across the smallest gap; the coarse
    one cells_per_gap); max_cells is only a safety cap on the merged panel count (h doubles until it fits). The two solves
    run concurrently when more than one thread is allowed. Returns (C, total panels, h of the fine mesh, info dict)."""
    bbox = shapely.union_all(metals).bounds
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    gap = min(shapely.distance(metals[i], metals[j]) for i in range(len(metals)) for j in range(i + 1, len(metals)))
    hf = max(gap / (2 * cells_per_gap), size / 8192)
    if not rich:
        C, n, h, n_raw = _cap_flagship(metals, eps_r, 2 * hf, max_cells, grade, lmax_mult, merge_grade)
        return C, n, h, {"h_target_um": 2 * hf, "min_gap_um": gap, "cap_hit": h > 2 * hf * 1.0001, "raw_cells": n_raw, "extrapolated": False}
    rf, rc = _both(partial(_cap_flagship, metals, eps_r, hf, max_cells, grade, lmax_mult, merge_grade),
                   partial(_cap_flagship, metals, eps_r, 2 * hf, max_cells, grade, lmax_mult, merge_grade))
    for r in (rf, rc):
        if isinstance(r, np.linalg.LinAlgError):
            raise r
    (Cf, nf, h_f, raw_f), (Cc, nc, h_c, raw_c) = rf, rc
    r = h_c / h_f
    C = Cf if r < 1.01 else Cf + (Cf - Cc) / (r - 1)
    info = {"h_target_um": hf, "min_gap_um": gap, "cap_hit": h_f > hf * 1.0001, "raw_cells": raw_f, "coarse_cells": nc,
            "extrapolated": bool(r >= 1.01)}
    return C, nf + nc, h_f, info
