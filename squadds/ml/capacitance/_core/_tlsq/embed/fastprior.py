"""Local (windowed) MoM prior: PRIOR_SCHEME.md Part III prototype.

The Maxwell entry C_ij is defined with every other conductor at 0 V, and a ground plane screens the field, so the
entries of a conductor are set by its neighbourhood. The layout is partitioned into windows (recursive bisection of
the conductor sites until the metal edge length inside each window fits a panel budget); a window models its core
conductors (whose matrix rows it provides), the halo conductors within R_g of the core as separate nets at 0 V, the
ground plane cropped by a margin around core + halo, and every farther conductor filled into the ground. Each window
is a strip-merged conductor-anchored mesh (the flagship mesh family, PRIOR_SCHEME.md Part II) solved densely with the
exact near field and one right-hand side per core conductor, optionally Richardson-extrapolated (h and 2 h). Windows
that are translation copies of one another are solved once (geometric hash). Cost: O(number of distinct windows) x
(a bounded dense solve), independent of the chip size; a layout that fits one window is one global solve.
Pairs never co-windowed (farther than R_g apart) are not predicted: NaN in C and True in far_mask.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import shapely
from scipy.spatial import cKDTree

from ..geometry import sample_boundary
from ..uge.blocks import _kratio
from . import proxy as P

PANELS_PER_EDGE_H = 0.75  # merged panels per (um of metal edge / h), measured on grid_qubit_1 (18.6k at h = 2 um, 50 mm of edges)


def _valid(g):
    g = shapely.make_valid(g)
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def gap_of(metals: list) -> float:
    tree = shapely.STRtree(metals)
    d = np.inf
    for i, g in enumerate(metals):
        j, dist = tree.query_nearest(g, return_distance=True, exclusive=True, all_matches=False)
        if len(dist):
            d = min(d, float(dist[0]))
    return d


def within(geoms: list, targets: list, R: float) -> np.ndarray:
    """Boolean (len(geoms),): geometry within distance R of any target (STRtree, O(n log n))."""
    out = np.zeros(len(geoms), bool)
    if not targets or not geoms:
        return out
    tree = shapely.STRtree(geoms)
    for t in targets:
        out[tree.query(t, predicate="dwithin", distance=R)] = True
    return out


def plan_windows(G, conds: list, h: float, R_g: float, margin: float, panel_budget: int, global_budget: int | None = None,
                 split_gain: float = 0.75, level_rings: tuple = ()) -> list:
    """Windows (dicts: core, halo, box). One global window (every conductor in the core, no halo, no crop) when the
    metal edge length of the whole layout fits global_budget panels; otherwise recursive bisection of the conductor
    sites (bbox centres) until the estimated merged panel count of the window fits panel_budget. The estimate weights
    the metal edge length by the grading level it will be meshed at (1, 1/2, 1/4, ... beyond each of level_rings um
    from the core; ground edges beyond R_g of the members are filled and count 0), so the budget measures the core
    content, not the fixed halo + margin overhead. Each split is snapped to the widest coordinate gap between sites in
    the central half of the sorted order (cores follow the natural cell structure); a split whose child is not below
    split_gain x the parent (a dense cluster) is not made and the window is solved whole (H-matrix above dense_max)."""
    sites = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in shapely.bounds(np.array(conds, dtype=object))])
    per_panel = h / PANELS_PER_EDGE_H
    all_idx = np.arange(len(conds))
    L_all = G.length + sum(c.length for c in conds)
    if L_all <= (global_budget if global_budget is not None else panel_budget) * per_panel:
        return [{"core": all_idx, "halo": np.zeros(0, int), "box": None, "n_core": len(conds), "n_halo": 0}]
    out = []
    Gb = G.boundary
    gtree = shapely.STRtree(list(getattr(Gb, "geoms", [Gb])))
    rings = list(level_rings) + [np.inf]

    def window_of(core):
        halo = within(conds, [conds[i] for i in core], R_g)
        halo[core] = False
        halo = np.nonzero(halo)[0]
        x0, y0, x1, y1 = shapely.union_all([conds[i] for i in np.concatenate([core, halo])]).bounds
        box = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        return halo, box

    def est_panels(core, halo, box):
        cg = shapely.union_all([conds[i] for i in core])
        members = shapely.union_all([conds[i] for i in np.concatenate([core, halo])])
        b = shapely.box(*box)
        gl = shapely.union_all([gtree.geometries[k] for k in gtree.query(b)]) if len(gtree.geometries) else Gb
        edges = shapely.union_all([shapely.intersection(gl, b).intersection(members.buffer(R_g)), members.boundary])
        L, prev, w = 0.0, None, 1.0
        for r in rings:
            ring = cg.buffer(r) if np.isfinite(r) else b
            if prev is not None:
                ring = ring.difference(prev)
            L += w * shapely.intersection(edges, ring).length
            prev, w = (cg.buffer(r) if np.isfinite(r) else b), w / 2
        return L / per_panel

    def rec(idx, n_parent=None):
        halo, box = window_of(idx)
        n = est_panels(idx, halo, box)
        if len(idx) == 1 or n <= panel_budget or (n_parent is not None and n > split_gain * n_parent):
            out.append({"core": idx, "halo": halo, "box": box, "est_panels": n})
            return
        s = sites[idx]
        ax = int(np.argmax(s.max(0) - s.min(0)))
        order = idx[np.argsort(s[:, ax], kind="stable")]
        x = sites[order, ax]
        lo = max(1, len(order) // 4)
        hi = max(lo + 1, min(len(order), (3 * len(order)) // 4 + 1))
        gaps = x[lo:hi] - x[lo - 1:hi - 1]
        m = lo + int(np.argmax(gaps))
        rec(order[:m], n), rec(order[m:], n)

    rec(all_idx)
    for w in out:
        w["n_core"], w["n_halo"] = len(w["core"]), len(w["halo"])
    return out


def window_metals(G, conds: list, w: dict, fill_um: float = 250.0, keep_um: float = 0.0) -> list:
    """[ground, core..., halo...] of window w: ground = chip ground cropped to the box plus every farther conductor
    inside the box as ground metal (0 V), whose etched surroundings within fill_um are filled in where they are more
    than keep_um from any member (their edges leave the mesh; nothing changes within keep_um of a member)."""
    members = np.concatenate([w["core"], w["halo"]]).astype(int)
    if w["box"] is None:
        return [G] + [conds[i] for i in members]
    box = shapely.box(*w["box"])
    far = np.setdiff1d(np.nonzero(within(conds, [box], 0.0))[0], members)
    Gw = shapely.intersection(G, box)
    if len(far):
        chip = shapely.box(*G.bounds)
        farg = shapely.union_all([conds[i] for i in far])
        fill = shapely.intersection(shapely.intersection(chip.difference(G), farg.buffer(fill_um)), box)
        fill = fill.difference(shapely.union_all([conds[i] for i in members]).buffer(keep_um))
        Gw = shapely.union_all([Gw, shapely.intersection(farg, box), fill])
    return [_valid(Gw)] + [conds[i] for i in members]


def anchored_cells_graded(metals: list, h_f: float, core, levels: tuple, grade: float = 2.0, majority: bool = False):
    """`proxy.anchored_cells` with the finest panel graded by the distance to the core conductors: h_f within levels[0]
    of `core`, 2 h_f up to levels[1], 4 h_f up to levels[2], ... (same lattice; with levels=() identical to the plain
    mesh). Returns centres (n, 2), sizes (n, 2), owner (n,)."""
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
    gq = P._Geoms(list(metals) + [edges, core])
    lv = np.asarray(levels, float)
    tst, inn = np.ones((len(metals), len(xs)), bool), np.zeros((len(metals), len(xs)), bool)
    out_c, out_h, out_o = [], [], []
    while len(xs):
        boxes = shapely.box(xs, ys, xs + h, ys + h)
        cov, hit, inn, pcs = P._cover(gq, boxes, h, tst, inn)
        keep = cov.sum(0) > 1e-3
        xs, ys, h, cov, boxes, hit, inn = xs[keep], ys[keep], h[keep], cov[:, keep], boxes[keep], hit[:, keep], inn[:, keep]
        pcs = [p[keep] for p in pcs]
        cx, cy = xs + h / 2, ys + h / 2
        full = cov.max(0) > 0.999
        if len(lv):
            maybe = h <= h_f * 2 ** len(lv) * 1.0001  # cells that can be at their last level
            hl = np.zeros(len(xs))
            if maybe.any():
                dc = P._distance(gq, len(metals) + 1, boxes[maybe])
                hl[maybe] = h_f * 2.0 ** np.searchsorted(lv, dc, side="right")
            last = h <= hl * 1.0001
        else:
            last = h <= h_f * 1.0001
        d = np.full(len(xs), np.inf)
        far = full & ~last
        d[far] = P._distance(gq, len(metals), boxes[far])
        leaf = full & (last | (h <= grade * d + 1e-12))
        out_c.append(np.stack([cx[leaf], cy[leaf]], 1))
        out_h.append(np.stack([h[leaf], h[leaf]], 1))
        out_o.append(cov[:, leaf].argmax(0))
        cut = last & ~leaf
        if majority and cut.any():
            take = cut & (cov.max(0) >= 0.5)
            out_c.append(np.stack([cx[take], cy[take]], 1))
            out_h.append(np.stack([h[take], h[take]], 1))
            out_o.append(cov[:, take].argmax(0))
            cut = cut & False
        if cut.any():
            gs = gq.get()
            for k, g in enumerate(gs[:len(metals)]):
                shapely.prepare(g)
                sel = np.nonzero(cut & (cov[k] > 1e-3))[0]
                if not len(sel):
                    continue
                pcs_k = pcs[k][sel]
                bb = shapely.bounds(pcs_k)
                cen = shapely.get_coordinates(shapely.centroid(pcs_k))
                inside = shapely.contains_xy(g, cen[:, 0], cen[:, 1])
                if not inside.all():
                    cen[~inside] = shapely.get_coordinates(shapely.point_on_surface(pcs_k[~inside]))
                out_c.append(cen)
                out_h.append(np.maximum(np.stack([bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]], 1), 1e-6))
                out_o.append(np.full(len(sel), k))
            gq.put(gs)
        split = ~leaf & ~last
        hs = h[split] / 2
        xs = np.concatenate([xs[split], xs[split] + hs, xs[split], xs[split] + hs])
        ys = np.concatenate([ys[split], ys[split], ys[split] + hs, ys[split] + hs])
        h = np.concatenate([hs] * 4)
        tst, inn = np.tile((hit & ~inn)[:, split], 4), np.tile(inn[:, split], 4)
    return np.concatenate(out_c), np.concatenate(out_h), np.concatenate(out_o)


def strip_merge(c, wh, o, metals, h_f: float, lmax_mult: float = 16.0, grade: float = 1.0):
    """`proxy.strip_merge` with the strip length bounded by lmax_mult x the panel's own lattice level (max(width across
    the edge, h_f), so coarser regions get proportionally longer strips) and by grade x the distance to the nearest
    polygon vertex (coarse square cells therefore merge into longer strips than in proxy.strip_merge)."""
    lines, S, verts = P._segments(metals)
    near = shapely.STRtree(lines).nearest(shapely.points(c))
    d = S[near, 1] - S[near, 0]
    horiz = np.abs(d[:, 0]) >= np.abs(d[:, 1])
    dv, _ = P.cKDTree(verts).query(c, workers=-1)
    allow = np.minimum(lmax_mult * np.maximum(wh.min(1), h_f), grade * dv)
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


def solve_window_rows(metals: list, n_core: int, h: float, eps_r: float = 11.45, lmax_mult: float = 16.0, merge_grade: float = 1.0,
                      grade: float = 2.0, rich: bool = True, dense_max: int = 12000, levels: tuple = ()) -> tuple:
    """Rows of the Maxwell matrix for the core conductors (metals[1:1+n_core]): (rows (n_core, len(metals)), info).
    levels: distances (um) from the core beyond which the finest panel doubles (core-graded mesh); () = uniform."""
    info = {}
    core = shapely.union_all(metals[1:1 + n_core])

    def one(hh, tag, majority=False):
        t = time.perf_counter()
        raw = anchored_cells_graded(metals, hh, core, levels, grade, majority=majority)
        c, wh, o = strip_merge(*raw, metals, hh, lmax_mult, merge_grade)
        info[f"mesh_s_{tag}"], info[f"raw_{tag}"], info[f"panels_{tag}"] = time.perf_counter() - t, len(raw[0]), len(c)
        info[f"share_{tag}"] = [float(np.mean(o == 0)), float(np.mean((o >= 1) & (o <= n_core))), float(np.mean(o > n_core))]
        t = time.perf_counter()
        B = (o[:, None] == np.arange(1, 1 + n_core)[None, :]).astype(float)
        if len(c) <= dense_max:
            A = P._assemble(c, wh, P.rect_self(wh[:, 0], wh[:, 1]), eps_r, P.rect_potential)
            info[f"assemble_s_{tag}"] = time.perf_counter() - t
            t = time.perf_counter()
            try:
                Q = P._chol_solve(A, B)
            except np.linalg.LinAlgError:  # cut panels of different nets overlap: whole finest cells by majority (as the flagship)
                if majority:
                    raise
                del A
                info[f"majority_{tag}"] = True
                return one(hh, tag, majority=True)
            info[f"solver_{tag}"] = "dense"
        else:
            H = P._HMatrix(c, wh, eps_r, P.HMAT["leaf"], P.HMAT["eta"], P.HMAT["eps"])
            info[f"assemble_s_{tag}"] = time.perf_counter() - t
            t = time.perf_counter()
            Xp, _, _ = H.pcg(B[H.perm], P.HMAT["tol"], P.HMAT["maxit"])
            Q = np.empty_like(Xp)
            Q[H.perm] = Xp
            info[f"solver_{tag}"] = "H-matrix PCG"
        info[f"solve_s_{tag}"] = time.perf_counter() - t
        return np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)  # (n_core, n_metals): charge on a with core k at 1 V

    Rf = one(h, "f")
    if not rich:
        return Rf, info
    Rc = one(2 * h, "c")
    return 2 * Rf - Rc, info


def canonical_hash(metals: list, n_core: int, h: float) -> str:
    """Translation-invariant key of a window (lattice anchored to the signal bbox, coordinates rounded to 0.1 nm)."""
    x0, y0 = shapely.union_all(metals[1:]).bounds[:2]
    tr = [shapely.transform(g, lambda xy: np.round(xy - [x0, y0], 4)) for g in metals]
    hsh = hashlib.sha1(f"{n_core}:{h:.9g}:".encode())
    for g in tr:
        hsh.update(shapely.to_wkb(g))
    return hsh.hexdigest()


def _job(args):
    wi, metals, n_core, h, kw = args
    rows, info = solve_window_rows(metals, n_core, h, **kw)
    return wi, rows, info


def local_prior(G, conds: list, eps_r: float = 11.45, cells_per_gap: float = 3.0, rich: bool = True, panel_budget: int = 12000,
                halo_gaps: float = 100.0, margin_gaps: float = 100.0, margin_min_um: float = 150.0, fill_um: float = 250.0,
                lmax_mult: float = 16.0, merge_grade: float = 1.0, workers: int = 1, cache: bool = True, dense_max: int = 12000,
                global_budget: int | None = None, level_gaps: tuple = ()) -> tuple:
    """Local prior of a layout (ground G, conductors conds, never merged): (C (N+1, N+1) fF with NaN for far pairs,
    far_mask, info). Finest panel h = gap / cells_per_gap; with rich=True the Richardson pair is (h, 2 h). Windows
    are solved in parallel on `workers` processes."""
    t_all = time.perf_counter()
    metals_all = [G] + list(conds)
    gap = gap_of(metals_all)
    h = gap / cells_per_gap
    R_g, margin = halo_gaps * gap, max(margin_gaps * gap, margin_min_um)
    t = time.perf_counter()
    wins = plan_windows(G, conds, h, R_g, margin, panel_budget, global_budget, level_rings=tuple(float(g) * gap for g in level_gaps))
    t_plan = time.perf_counter() - t
    t = time.perf_counter()
    mats = [window_metals(G, conds, w, fill_um, keep_um=R_g) for w in wins]
    keys = [canonical_hash(m, w["n_core"], h) if cache else str(i) for i, (m, w) in enumerate(zip(mats, wins))]
    t_build = time.perf_counter() - t
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, i)
    kw = dict(eps_r=eps_r, lmax_mult=lmax_mult, merge_grade=merge_grade, rich=rich, dense_max=dense_max,
              levels=tuple(float(g) * gap for g in level_gaps))
    jobs = [(i, mats[i], wins[i]["n_core"], h, kw) for i in uniq.values()]
    t = time.perf_counter()
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
            res = {wi: (rows, info) for wi, rows, info in ex.map(_job, jobs)}
    else:
        res = {wi: (rows, info) for wi, rows, info in map(_job, jobs)}
    t_solve = time.perf_counter() - t
    N = len(conds) + 1
    acc, cnt = np.zeros((N, N)), np.zeros((N, N))
    for i, w in enumerate(wins):
        rows, _ = res[uniq[keys[i]]]
        idx = np.concatenate([[0], w["core"] + 1, w["halo"] + 1]).astype(int)
        for k, ci in enumerate(w["core"] + 1):
            acc[ci, idx] += rows[k]
            cnt[ci, idx] += 1
    C = np.full((N, N), np.nan)
    have = cnt > 0
    C[have] = acc[have] / cnt[have]
    # symmetrise: mean of the two estimates when both exist, the one available otherwise; ground row from the columns
    S = np.where(have & have.T, 0.5 * (np.nan_to_num(C) + np.nan_to_num(C.T)), np.where(have, C, C.T))
    S[0, 0] = np.nan
    far = ~(have | have.T)
    np.fill_diagonal(far, False)
    infos = [res[j][1] for j in uniq.values()]
    info = {"n_windows": len(wins), "n_unique": len(uniq), "h_um": h, "min_gap_um": gap, "R_g_um": R_g, "margin_um": margin,
            "panels_f": [i["panels_f"] for i in infos], "est_panels": [w.get("est_panels", 0) for w in wins], "raw_f": [i["raw_f"] for i in infos],
            "panels_c": [i.get("panels_c", 0) for i in infos], "cores": [w["n_core"] for w in wins], "halos": [w["n_halo"] for w in wins],
            "solvers": sorted({i["solver_f"] for i in infos}), "n_far_pairs": int(far[np.triu_indices(N, 1)].sum()),
            "mesh_s": sum(i["mesh_s_f"] + i.get("mesh_s_c", 0.0) for i in infos),
            "assemble_s": sum(i["assemble_s_f"] + i.get("assemble_s_c", 0.0) for i in infos),
            "solve_s": sum(i["solve_s_f"] + i.get("solve_s_c", 0.0) for i in infos),
            "level_gaps": list(level_gaps), "plan_s": t_plan, "build_s": t_build, "windows_wall_s": t_solve, "total_s": time.perf_counter() - t_all, "rich": rich}
    return S, far, info


# ---------------------------------------------------------------- flagship-fast: the cheap prior (PRIOR_SCHEME.md Part III, sections 12 and 16)
# Hybrid H = micro-MoM (`micro_mom`: the flagship mesh family at a fixed panel budget per window, dense Cholesky) for self,
# ground and co-windowed pairs + the closed-form edge-coupling prior (`formula_prior`) for pairs never co-windowed.
# Mirrored verbatim (function sources) in deploy/tlsq_infer/src/tlsq_infer/_vendor/fastprior.py.

# formula constants: frozen 2026-10-09, not tuned on labels
FORMULA_STEP_UM = 0.25  # boundary sample step (as the pair tokens)
FORMULA_R_GAPS = 150.0  # pairs farther apart than this (in smallest gaps) use the bounding-box form
FORMULA_SCREEN_UM = 1.0  # the screening factor is evaluated every FORMULA_SCREEN_UM of boundary and held in between
LITE_REFINE = 3  # rejected window solves: refine h by 2 up to this many times before majority-owned cells
LITE_LEVEL_GAPS = (10.0, 20.0, 40.0, 80.0)  # windows of large layouts: finest panel within 10 gaps of the core, doubling beyond 10/20/40/80
LITE_WINDOW_BUDGET = 2.0  # windows of large layouts: panel budget = this x the budget of the one-window (component) prior
LITE_WINDOW_PPG = [None]  # windows of large layouts: panels per smallest gap of the finest level instead of a budget (None = budget; PRIOR_SCHEME.md 14.4)
LITE_MAX_CORE = [None]  # windows of large layouts: conductors per core (None = the max_core argument; PRIOR_SCHEME.md 14.4.2)
LITE_GROWTH = 2.0  # merged panels per halving of h (measured 1.6-2.2 on the training families)
ALPHA_GAP_SHARE = 0.52  # dipole far pairs: perimeter x gap band share of the effective area; fit 0.519 (median log-ratio 0 vs the global gap/3 MoM, 68 cross-pocket pairs > 50 gaps of grid_qubit_1; PRIOR_SCHEME.md 14.2)
DIPOLE_K = 8  # nearest boundary samples searched for the nearest other conductor (gbar)
ROW_SUM_EPS = 0.0143  # training median of 1 - sum_j |C_ij| / C_ii (self = row sum / (1 - eps))
EPS0_FF_UM = 8.8541878128e-3


def plan_cores(sites: np.ndarray, max_core: int = 8) -> list:
    """Cores (index arrays) by recursive bisection of the conductor sites (n, 2) along the longer extent until each core
    has <= max_core conductors; each split is snapped to the widest coordinate gap between sites in the central half
    of the sorted order (ties broken by the other coordinate, so the cores do not depend on the conductor order)."""
    out = []

    def rec(idx):
        if len(idx) <= max_core:
            out.append(idx)
            return
        s = sites[idx]
        ax = int(np.argmax(s.max(0) - s.min(0)))
        order = idx[np.lexsort((s[:, 1 - ax], s[:, ax]))]
        x = sites[order, ax]
        lo = max(1, len(order) // 4)
        hi = max(lo + 1, min(len(order), (3 * len(order)) // 4 + 1))
        m = lo + int(np.argmax(x[lo:hi] - x[lo - 1:hi - 1]))
        rec(order[:m]), rec(order[m:])

    rec(np.arange(len(sites)))
    return out


def _rot90(k: int):
    """Exact rotation of coordinates by k x 90 degrees (swaps and sign flips only)."""
    def f(xy):
        x, y = xy[:, 0], xy[:, 1]
        return np.stack([(x, y), (-y, x), (-x, -y), (y, -x)][k % 4], 1)
    return f


def canonical_frames(G, conds: list) -> list:
    """The layout in canonical frames, so that the lattice-anchored mesh (and hence the prior) does not depend on the
    position, the 90-degree orientation or the conductor order of the input: rotations by k x 90 degrees whose
    rotation-invariant key (signal bounding box, sorted conductor centroids and convex-hull vertices about the box
    centre, ground bounds, rounded to 1 nm) is smallest (several when the layout is symmetric), translation of the signal
    bounding box corner to the origin, coordinates rounded to 1e-6 um, collinear vertices removed (`shapely.simplify(0)`),
    rings in normal form (`shapely.normalize`), conductors sorted by (centroid, area). Returns [(G, conds, order)]:
    conds[k] of a frame is input conductor order[k]."""
    keys = []
    for k in range(4):
        f = _rot90(k)
        sig = [shapely.transform(c, f) for c in conds]
        b = shapely.bounds(np.array(sig, dtype=object))
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        cen = shapely.get_coordinates(shapely.centroid(np.array(sig, dtype=object))) - [(x0 + x1) / 2, (y0 + y1) / 2]
        hull = shapely.get_coordinates(shapely.convex_hull(shapely.union_all(sig)))[:-1] - [(x0 + x1) / 2, (y0 + y1) / 2]
        gb = np.round(np.array(shapely.transform(G, f).bounds) - [(x0 + x1) / 2, (y0 + y1) / 2] * 2, 3)
        key = ((round(x1 - x0, 3), round(y1 - y0, 3)) + tuple(sorted(map(tuple, np.round(cen, 3).tolist())))
               + tuple(sorted(map(tuple, np.round(hull, 3).tolist()))) + tuple(gb.tolist()))
        keys.append((key, k, x0, y0))
    best = min(kk[0] for kk in keys)
    out = []
    for key, k, x0, y0 in keys:
        if key != best:
            continue
        f = _rot90(k)

        def to(g, f=f, x0=x0, y0=y0):
            return shapely.normalize(shapely.simplify(shapely.transform(g, lambda xy: np.round(f(xy) - [x0, y0], 6)), 0.0))

        cs = [to(c) for c in conds]
        cen = shapely.get_coordinates(shapely.centroid(np.array(cs, dtype=object)))
        order = np.lexsort((np.round([c.area for c in cs], 6), np.round(cen[:, 1], 6), np.round(cen[:, 0], 6)))
        out.append((to(G), [cs[i] for i in order], order))
    return out


def _gap_size(metals: list) -> tuple:
    """(smallest gap, layout size) exactly as mom_capacitance_gds_flagship computes them (all pairs; small layouts)."""
    bbox = shapely.union_all(metals).bounds
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    gap = min(shapely.distance(metals[i], metals[j]) for i in range(len(metals)) for j in range(i + 1, len(metals)))
    return gap, size


def _graded_mesh(metals: list, n_core: int, h: float, levels: tuple, majority: bool = False) -> tuple:
    """Core-graded flagship-family mesh (`anchored_cells_graded` + `strip_merge`): finest panel h within levels[0] of the
    core conductors (metals[1:1 + n_core]), doubling beyond each level. Returns (centres, sizes, owners, h, raw count)."""
    raw = anchored_cells_graded(metals, h, shapely.union_all(metals[1:1 + n_core]), levels, majority=majority)
    c, wh, o = strip_merge(*raw, metals, h)
    return c, wh, o, h, len(raw[0])


def _lite_mesh(metals: list, h: float, budget: float, n_core: int = 0, levels: tuple = ()) -> tuple:
    """Flagship-family mesh at the finest lattice level h 2^k predicted to fit the budget: a probe mesh at the level
    estimated (2 merged panels per h of metal edge) to hold ~budget / 4 panels, then the level LITE_GROWTH^j finer
    predicted to hold <= budget panels (each halving of h multiplies the panels by ~LITE_GROWTH), doubled while it does
    not fit. levels (um from the n_core core conductors): core-graded mesh (`_graded_mesh`); () = uniform
    (`proxy.mesh_flagship`). With an unbounded budget: the mesh at h. Returns (centres, sizes, owners, h, raw count)."""
    def mesh(hh):
        if not levels:
            return P.mesh_flagship(metals, hh, budget)
        while True:
            r = _graded_mesh(metals, n_core, hh, levels)
            if len(r[0]) <= budget:
                return r
            hh *= 2.0

    if not np.isfinite(budget):
        return mesh(h)
    L = sum(g.length for g in metals)
    k = max(0, int(np.ceil(np.log2(max(2.0 * L / (h * budget / 4), 1.0)))))
    probe = mesh(h * 2.0**k)
    j = min(int(np.floor(np.log(budget / len(probe[0])) / np.log(LITE_GROWTH))), int(round(np.log2(probe[3] / h))))
    return probe if j <= 0 else mesh(probe[3] / 2.0**j)


def _sym_solve(A, B):
    """Solve with the lower triangle of the Fortran-ordered A from `proxy._assemble`: Cholesky, or the symmetric
    indefinite (Bunch-Kaufman) factorisation when A is not positive definite."""
    try:
        return P._chol_solve(A.copy(order="F"), B)
    except np.linalg.LinAlgError:
        from scipy.linalg import solve
        return solve(A, B, lower=True, assume_a="sym", overwrite_a=True, check_finite=False)


def _window_solve(metals: list, n_core: int, h: float, budget: float, eps_r: float, all_rhs: bool, levels: tuple = ()) -> tuple:
    """Flagship-family mesh of one window at the lattice level of h that fits the budget (`_lite_mesh`), exact near
    field, dense Cholesky (H-matrix PCG above DENSE_MAX panels); right-hand sides: every metal (all_rhs, as
    `proxy._cap_flagship`) or the n_core core conductors; levels: core-graded mesh (`_lite_mesh`). A solve is rejected
    when the system is not positive definite, a metal has no panel or (all_rhs) a mutual entry is not negative (cut
    panels of different conductors overlapping on a lattice coarser than the features); the mesh is then refined (h / 2, budget lifted) up to LITE_REFINE times, and finally
    meshed with majority-owned finest cells (as the flagship). Returns (R (n_rhs, n_metals) charges per metal, info)."""
    cols = np.arange(len(metals)) if all_rhs else np.arange(1, 1 + n_core)
    t_mesh = t_asm = t_sol = 0.0
    hh, cap, refined, majority = h, budget, 0, False
    while True:
        t = time.perf_counter()
        if not refined:
            c, wh, o, hh, n_raw = _lite_mesh(metals, hh, cap, n_core, levels)
        elif levels:
            c, wh, o, hh, n_raw = _graded_mesh(metals, n_core, hh, levels, majority)
        else:
            c, wh, o, hh, n_raw = P.mesh_flagship(metals, hh, cap, majority=majority)
        t_mesh += time.perf_counter() - t
        t = time.perf_counter()
        try:
            if len(c) <= P.DENSE_MAX:
                A = P._assemble(c, wh, P.rect_self(wh[:, 0], wh[:, 1]), eps_r, P.rect_potential)
                t_asm += time.perf_counter() - t
                t = time.perf_counter()
                Bm = (o[:, None] == cols[None, :]).astype(float)
                if all_rhs:
                    Q = P._chol_solve(A, Bm)
                else:  # windows: an indefinite system (overlapping cut panels on the coarse graded levels) is solved by LDL^T
                    Q = _sym_solve(A, Bm)
            else:
                Q = P.mom_solve_fast(c, wh, o, len(metals), eps_r)[:, cols]
            R = np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)
            off = np.ones(R.shape, bool)
            off[np.arange(len(cols)), cols] = False
            ok = bool(np.all(np.bincount(o, minlength=len(metals)) > 0) and (not all_rhs or np.all(R[off] < 0)))
        except np.linalg.LinAlgError:
            ok = False
        t_sol += time.perf_counter() - t
        if ok or majority:
            break
        if refined < LITE_REFINE:
            hh, cap, refined = hh / 2, np.inf, refined + 1
        else:
            majority = True
    return R, {"h_um": hh, "panels": len(c), "raw": n_raw, "refined": refined, "majority": majority, "mesh_s": t_mesh,
               "assemble_s": t_asm, "solve_s": t_sol}


def _lite_window(G, conds: list, core, R_g: float, margin: float, fill_um: float, tree, etched) -> tuple:
    """Window of a core: ([ground, core..., halo...], halo). Halo = conductors within R_g of a core conductor (separate
    nets at 0 V); ground = chip ground clipped to the core + halo box widened by margin, plus every farther conductor in
    the box as ground metal with its etched surroundings within fill_um filled where farther than R_g from a member."""
    hit = tree.query([conds[i] for i in core], predicate="dwithin", distance=R_g)[1]
    halo = np.setdiff1d(np.unique(hit), core)
    members = np.concatenate([core, halo]).astype(int)
    b = shapely.bounds(np.array([conds[i] for i in members], dtype=object))
    box = (b[:, 0].min() - margin, b[:, 1].min() - margin, b[:, 2].max() + margin, b[:, 3].max() + margin)
    far = np.setdiff1d(tree.query(shapely.box(*box), predicate="intersects"), members)
    Gw = shapely.clip_by_rect(G, *box)
    if len(far):
        farg = shapely.clip_by_rect(shapely.union_all([conds[i] for i in far]), *box)
        fill = shapely.intersection(shapely.clip_by_rect(etched, *box), farg.buffer(fill_um, quad_segs=4))
        fill = fill.difference(shapely.union_all([conds[i] for i in members]).buffer(R_g, quad_segs=4))
        Gw = shapely.union_all([Gw, farg, fill])
    return [_valid(Gw)] + [conds[i] for i in members], halo


_LITE_BUDGET_USED: list = [0.0]
_LITE: dict = {}  # layout of the running micro_mom call, inherited by forked window workers


def _lite_job(k: int) -> tuple:
    L = _LITE
    t = time.perf_counter()
    metals, halo = _lite_window(L["G"], L["conds"], L["cores"][k], L["R_g"], L["margin"], L["fill_um"], L["tree"], L["etched"])
    t_build = time.perf_counter() - t
    bx = shapely.bounds(metals[0])
    h = max(L["h"], max(bx[2] - bx[0], bx[3] - bx[1]) / 4096)
    budget = L["budget"]
    if LITE_WINDOW_PPG[0]:  # resolution-controlled windows (PRIOR_SCHEME.md 14.4): finest panel gap / ppg, no budget
        h, budget = L["gap"] / LITE_WINDOW_PPG[0], np.inf
    R, info = _window_solve(metals, len(L["cores"][k]), h, budget, L["eps_r"], False, L["levels"])
    info.update(build_s=t_build, n_halo=len(halo), n_core=len(L["cores"][k]))
    return k, halo, R, info


def micro_mom(G, conds: list, eps_r: float = 11.45, budget: float = 1500, cells_per_gap: float = 3.0, max_core: int = 8,
              halo_gaps: float = 100.0, margin_gaps: float = 100.0, margin_min_um: float = 150.0, fill_um: float = 250.0,
              workers: int = 1, level_gaps: tuple = LITE_LEVEL_GAPS, window_budget: float | None = None, canonical: bool = True) -> tuple:
    """Cheap micro-MoM prior (PRIOR_SCHEME.md section 12, M / hybrid H): (C (N+1, N+1) fF Maxwell, far_mask, info).
    <= max_core conductors: one global window = the layout as given, h = max(gap / cells_per_gap, size / 4096) doubled
    on the flagship lattice until the merged mesh has <= budget panels, every metal excited (with budget = inf this is
    `proxy.mom_capacitance_gds_flagship(rich=False)` bit for bit; `_lite_mesh` picks the level). More conductors: cores of <= max_core by bisection,
    one window per core (halo within halo_gaps, ground clipped at margin_gaps), entry (i, j) = mean of the two window
    estimates when both exist (a non-negative window estimate is not used), ground row from the columns, C_00 = row sum /
    (1 - ROW_SUM_EPS); pairs never co-windowed (far_mask, > halo_gaps apart) in different ground pockets take the screened
    dipole value (`dipole_far`), in the same pocket the closed form (`formula_prior`, quadrature step one smallest gap). level_gaps: core-graded window
    meshes (finest panel within level_gaps[0] smallest gaps of the core, doubling beyond each level); window systems that
    are not positive definite are solved by LDL^T; window budget = window_budget or LITE_WINDOW_BUDGET x budget. Windows
    run on `workers` forked processes. canonical: solved in the `canonical_frames` (mean over them when the layout is
    symmetric; invariant to translation, 90-degree rotation and conductor order), returned in the input order."""
    t_all = time.perf_counter()
    if canonical:  # mean over the canonical frames (one unless the layout is symmetric), in the input order
        runs = []
        for Gc, cc, order in canonical_frames(G, conds):
            C, far, info = micro_mom(Gc, cc, eps_r, budget, cells_per_gap, max_core, halo_gaps, margin_gaps, margin_min_um, fill_um,
                                     workers, level_gaps, window_budget, canonical=False)
            inv = np.r_[0, np.argsort(order) + 1]
            runs.append((C[np.ix_(inv, inv)], far[np.ix_(inv, inv)], info))
        C = runs[0][0] if len(runs) == 1 else sum(r[0] for r in runs) / len(runs)
        far = np.logical_or.reduce([r[1] for r in runs])
        info = runs[0][2]
        for k in ("mesh_s", "assemble_s", "solve_s", "build_s", "formula_s"):
            info[k] = sum(r[2][k] for r in runs)
        info["n_frames"] = len(runs)
        info["total_s"] = time.perf_counter() - t_all
        return C, far, info
    metals = [G] + list(conds)
    N = len(metals)
    far = np.zeros((N, N), bool)
    if len(conds) <= max_core:
        gap, size = _gap_size(metals)
        h = max(gap / cells_per_gap, size / 4096)
        R, wi = _window_solve(metals, len(conds), h, budget, eps_r, True)
        C = 0.5 * (R + R.T)
        info = {"n_windows": 1, "min_gap_um": gap, "h_um": [wi["h_um"]], "panels": [wi["panels"]], "panels_per_gap": [gap / wi["h_um"]],
                "majority": [wi["majority"]], "refined": [wi["refined"]], "mesh_s": wi["mesh_s"], "assemble_s": wi["assemble_s"], "solve_s": wi["solve_s"],
                "build_s": 0.0, "plan_s": 0.0, "formula_s": 0.0, "n_far_pairs": 0, "budget": budget}
        info["total_s"] = time.perf_counter() - t_all
        return C, far, info
    t = time.perf_counter()
    gap = gap_of(metals)
    R_g, margin = halo_gaps * gap, max(margin_gaps * gap, margin_min_um)
    b = shapely.bounds(np.array(conds, dtype=object))
    cores = plan_cores(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1), LITE_MAX_CORE[0] or max_core)
    tree = shapely.STRtree(conds)
    etched = shapely.box(*G.bounds).difference(G)
    _LITE.clear()
    _LITE.update(G=G, conds=conds, cores=cores, R_g=R_g, margin=margin, fill_um=fill_um, tree=tree, etched=etched,
                 h=gap / cells_per_gap, gap=gap, budget=LITE_WINDOW_BUDGET * budget if window_budget is None else window_budget, eps_r=eps_r,
                 levels=tuple(float(x) * gap for x in level_gaps))
    _LITE_BUDGET_USED[:] = [_LITE["budget"]]
    t_plan = time.perf_counter() - t
    t = time.perf_counter()
    if workers > 1 and len(cores) > 1:
        with ProcessPoolExecutor(min(workers, len(cores)), mp_context=get_context("fork")) as ex:
            res = list(ex.map(_lite_job, range(len(cores))))
    else:
        res = [_lite_job(k) for k in range(len(cores))]
    t_win = time.perf_counter() - t
    _LITE.clear()
    acc, cnt = np.zeros((N, N)), np.zeros((N, N))
    for k, halo, R, _ in res:
        idx = np.concatenate([[0], cores[k] + 1, halo + 1]).astype(int)
        for r, ci in enumerate(cores[k] + 1):
            use = (R[r] < 0) | (idx == ci)  # a non-negative mutual estimate of a window (tiny far entries) is not used
            acc[ci, idx[use]] += R[r][use]
            cnt[ci, idx[use]] += 1
    have = cnt > 0
    Cw = np.where(have, acc / np.where(have, cnt, 1), 0.0)
    C = np.where(have & have.T, 0.5 * (Cw + Cw.T), np.where(have, Cw, Cw.T))
    far = ~(have | have.T)
    np.fill_diagonal(far, False)
    t = time.perf_counter()
    iu = np.nonzero(np.triu(far, 1))
    t_dip, n_dip = 0.0, 0
    if len(iu[0]):
        pid = np.r_[-10**9, pocket_ids(G, conds)]
        cross = pid[iu[0]] != pid[iu[1]]  # different ground pockets: screened dipole; same pocket: closed form
        if (~cross).any():
            F, _ = formula_prior(G, conds, eps_r, step_um=max(FORMULA_STEP_UM, gap), pairs=np.stack(iu, 1)[~cross], gap=gap, workers=workers)
            C[iu[0][~cross], iu[1][~cross]] = F[iu[0][~cross], iu[1][~cross]]
        if cross.any():
            t_dip = time.perf_counter()
            C[iu[0][cross], iu[1][cross]] = dipole_far(G, conds, np.stack(iu, 1)[cross], eps_r, gap)
            t_dip, n_dip = time.perf_counter() - t_dip, int(cross.sum())
        C[iu[1], iu[0]] = C[iu]
    t_form = time.perf_counter() - t
    C[0, 0] = np.abs(C[0, 1:]).sum() / (1 - ROW_SUM_EPS)
    infos = [r[3] for r in res]
    info = {"n_windows": len(cores), "min_gap_um": gap, "R_g_um": R_g, "margin_um": margin, "h_um": [i["h_um"] for i in infos],
            "panels": [i["panels"] for i in infos], "panels_per_gap": [gap / i["h_um"] for i in infos],
            "majority": [i["majority"] for i in infos], "refined": [i["refined"] for i in infos], "cores": [i["n_core"] for i in infos], "halos": [i["n_halo"] for i in infos],
            "mesh_s": sum(i["mesh_s"] for i in infos), "assemble_s": sum(i["assemble_s"] for i in infos),
            "solve_s": sum(i["solve_s"] for i in infos), "build_s": sum(i["build_s"] for i in infos), "plan_s": t_plan,
            "windows_wall_s": t_win, "formula_s": t_form, "n_far_pairs": int(len(iu[0])), "dipole_s": t_dip,
            "n_dipole_pairs": n_dip, "budget": _LITE_BUDGET_USED[0],
            "workers": workers, "level_gaps": list(level_gaps)}
    info["total_s"] = time.perf_counter() - t_all
    return C, far, info


def pocket_ids(G, conds: list) -> np.ndarray:
    """Ground pocket of every conductor: index of the interior ring of the ground metal that contains it (STRtree of the
    rings' polygons, `within`); a conductor in no ring gets its own id (-1 - k)."""
    rings = [shapely.Polygon(r) for p in getattr(G, "geoms", [G]) for r in p.interiors]
    out = -1 - np.arange(len(conds))
    if rings:
        c, r = shapely.STRtree(rings).query(list(conds), predicate="within")
        out[c] = r
    return out


def _edge_samples(geoms: list, step: float) -> tuple:
    """Points every <= step along every polygon edge of geoms (vertices included), with the index of their geometry:
    (points (n, 2), owner (n,)). Vectorised over all edges (no per-ring interpolation)."""
    segs, sown = [], []
    for k, g in enumerate(geoms):
        for p in getattr(g, "geoms", [g]):
            for ring in [p.exterior, *p.interiors]:
                xy = np.asarray(ring.coords)
                segs.append(np.stack([xy[:-1], xy[1:]], 1))
                sown.append(np.full(len(xy) - 1, k))
    S, sown = np.concatenate(segs), np.concatenate(sown)
    L = np.sqrt(((S[:, 1] - S[:, 0]) ** 2).sum(1))
    n = np.maximum(1, np.ceil(L / step).astype(int))
    seg = np.repeat(np.arange(len(S)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return S[seg, 0] + t[:, None] * (S[seg, 1] - S[seg, 0]), sown[seg]


def dipole_far(G, conds: list, pairs, eps_r: float = 11.45, gap: float | None = None) -> np.ndarray:
    """Screened-dipole far couplings (PRIOR_SCHEME.md 14.2): for metal-index pairs (k, 2) (1 = first conductor),
    C_ij = -(1 + eps_r) eps0 At_i At_j / (2 pi d_eff^3), d_eff^2 = d^2 + a_i^2 + a_j^2, a = sqrt(At / pi), d = centroid
    distance, At = A + ALPHA_GAP_SHARE x P x gbar with gbar the mean distance from i's edge samples (step one gap) to the
    nearest other metal: ground edge samples (step gap / 4) by one k-d tree, other conductors by one k-d tree of all
    conductor samples with owner labels (DIPOLE_K neighbours). Returns the values (k,)."""
    metals = [G] + list(conds)
    gap = gap_of(metals) if gap is None else gap
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    pts, own = _edge_samples(list(conds), gap)
    own = own + 1
    sel = np.isin(own, np.unique(pairs))
    q = pts[sel]
    dg, _ = cKDTree(_edge_samples([G], gap / 4)[0]).query(q)
    d, idx = cKDTree(pts).query(q, k=min(DIPOLE_K, len(own)))
    if d.ndim == 1:
        d, idx = d[:, None], idx[:, None]
    d = np.where(own[np.minimum(idx, len(own) - 1)] != own[sel][:, None], d, np.inf).min(1)
    gbar = np.bincount(own[sel], np.minimum(dg, d), len(metals)) / np.maximum(np.bincount(own[sel], minlength=len(metals)), 1)
    area = np.array([g.area for g in metals])
    perim = np.array([g.length for g in metals])
    At = area + ALPHA_GAP_SHARE * perim * gbar
    cen = shapely.get_coordinates(shapely.centroid(np.array(metals, dtype=object)))
    a, b = pairs[:, 0], pairs[:, 1]
    d2 = ((cen[a] - cen[b]) ** 2).sum(1) + (At[a] + At[b]) / np.pi
    return -(1 + eps_r) * EPS0_FF_UM * At[a] * At[b] / (2 * np.pi * d2 ** 1.5)


def _ground_segments(G):
    """Edge segments (m, 2, 2) of the ground metal and an STRtree over them."""
    _, S, _ = P._segments([G])
    return S, shapely.STRtree(shapely.linestrings(S))


def _metal_length(A: np.ndarray, B: np.ndarray, S: np.ndarray, tree) -> np.ndarray:
    """Length of ground metal crossed by each segment A[k] -> B[k] (endpoints off the metal): crossings with the ground
    edges (STRtree) sorted along the segment, alternate inside / outside intervals summed."""
    out = np.zeros(len(A))
    if not len(A):
        return out
    li, gi = tree.query(shapely.linestrings(np.stack([A, B], 1)), predicate="intersects")
    if not len(li):
        return out
    p, r = A[li], B[li] - A[li]
    q, s = S[gi, 0], S[gi, 1] - S[gi, 0]
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(den) > 1e-12
    li, t = li[ok], np.clip(((q - p)[ok, 0] * s[ok, 1] - (q - p)[ok, 1] * s[ok, 0]) / den[ok], 0.0, 1.0)
    order = np.lexsort((t, li))
    li, t = li[order], t[order]
    start = np.r_[0, np.nonzero(np.diff(li))[0] + 1]
    n = np.diff(np.r_[start, len(li)])
    rank = np.arange(len(li)) - np.repeat(start, n)
    keep = rank < np.repeat(n // 2 * 2, n)
    sgn = np.where(rank % 2 == 1, 1.0, -1.0)
    out += np.bincount(li[keep], weights=(sgn * t)[keep], minlength=len(A))
    return out * np.sqrt(((B - A) ** 2).sum(1))


def _cps_m(d, w1, w2):
    """Coplanar-strip modulus (as `blocks.edge_features`; w = inf for the ground)."""
    if np.isinf(w1) or np.isinf(w2):
        wf = w2 if np.isinf(w1) else w1
        return d / (d + wf)
    return d * (d + w1 + w2) / ((d + w1) * (d + w2))


_FORM: dict = {}  # samples and trees of the running formula_prior call, inherited by forked row workers


def _formula_row(i: int) -> tuple:
    """Row i of the directed sums I[i, j] of `formula_prior` (j = the near metals of i; ground last term included)."""
    F = _FORM
    pts, kd, near, width, R, k, step_um, every = F["pts"], F["kd"], F["near"], F["width"], F["R"], F["k"], F["step"], F["every"]
    I = np.zeros(near.shape[0])
    dmin_other, dG = np.full(len(pts[i]), np.inf), None
    for j in np.nonzero(near[i])[0]:
        d, idx = kd[j].query(pts[i], distance_upper_bound=R if j > 0 else np.inf)
        ok = np.isfinite(d)
        if j > 0:
            dmin_other = np.minimum(dmin_other, d)
        else:
            dG = d
        if not ok.any():
            continue
        dd = np.maximum(d[ok], 0.05)
        if j == 0:
            S = np.ones(ok.sum())
        else:
            sel = np.arange(0, ok.sum(), every)
            A = pts[i][ok][sel]
            lg = _metal_length(A, pts[j][idx[ok][sel]], F["SG"], F["gtree"])
            S = np.exp(-lg / (dd[sel] + min(width[i], width[j])))[np.arange(ok.sum()) // every]
        I[j] = k * np.sum(step_um * _kratio(_cps_m(dd, width[i], width[j])) * S)
    if dG is not None:
        share = float(np.mean(dG <= dmin_other))
        I[0] += 8 * EPS0_FF_UM * (1 + F["eps_r"]) / 2 * np.sqrt(F["area"][i] / np.pi) * share
    return i, I


def formula_prior(G, conds: list, eps_r: float = 11.45, step_um: float = FORMULA_STEP_UM, R_gaps: float = FORMULA_R_GAPS,
                  pairs=None, gap: float | None = None, screen_um: float = FORMULA_SCREEN_UM, workers: int = 1,
                  max_core: int = 8) -> tuple:
    """Closed-form edge-coupling prior (PRIOR_SCHEME.md section 12, F): (C (N+1, N+1) fF Maxwell, info). No mesh, no solve.
    C_ij = eps0 (1 + eps_r) / 2 x mean over the directions (i, j), (j, i) of sum_p K(k') / K(k) S(p) step over the boundary
    samples p of i (modulus of the local widths 2 A / P and the distance to the nearest sample of j, within R_gaps
    smallest gaps; the ground without a bound), S(p) = exp(-l_G / (d + w_min)) with l_G the ground metal crossed on p -> nearest point of j. C_iG: the
    same sum against the ground edge (S = 1) + 8 eps0 eps_eff sqrt(A_i / pi) x (share of i's samples nearest to the
    ground). Pairs farther than R_gaps: the same form on the bounding boxes (facing length (P_i + P_j) / 2, distance of
    the boxes, l_G = ground area fraction x distance). Self terms: row sums / (1 - ROW_SUM_EPS). pairs (k, 2) of metal
    indices (0 = ground): only those mutual entries (the rest 0, no self terms). More than max_core conductors: the
    quadrature step is at least gap / 4 and the screening spacing at least one gap (numerically equivalent: median
    0.02 %, max 1.2 % on the grids vs 0.25 um), rows on `workers` forked processes."""
    t0 = time.perf_counter()
    metals = [G] + list(conds)
    N = len(metals)
    gap = gap_of(metals) if gap is None else gap
    if len(conds) > max_core:
        step_um, screen_um = max(step_um, gap / 4), max(screen_um, gap)
    R = R_gaps * gap
    k = EPS0_FF_UM * (1 + eps_r) / 2
    want = np.zeros((N, N), bool)
    if pairs is None:
        want[np.triu_indices(N, 1)] = True
    else:
        pairs = np.sort(np.asarray(pairs, int).reshape(-1, 2), 1)
        want[pairs[:, 0], pairs[:, 1]] = True
    area = np.array([g.area for g in metals])
    perim = np.array([g.length for g in metals])
    width = np.r_[np.inf, 2 * area[1:] / perim[1:]]
    tree = shapely.STRtree(list(conds))
    near = np.zeros((N, N), bool)
    a, b = tree.query(list(conds), predicate="dwithin", distance=R)
    near[a + 1, b + 1] = True
    near[0, 1:] = near[1:, 0] = True
    np.fill_diagonal(near, False)
    near &= want | want.T
    gx0, gy0, gx1, gy1 = G.bounds
    need = np.unique(np.nonzero(near)[0])
    pts, kd = {}, {}
    for i in np.unique(np.r_[need, np.nonzero(near.any(0))[0]]):
        p = sample_boundary(metals[i], step_um)
        if i == 0:  # ground edge = boundary of the etched region (not the chip outline)
            on = (np.abs(p[:, 0] - gx0) < 1e-6) | (np.abs(p[:, 0] - gx1) < 1e-6) | (np.abs(p[:, 1] - gy0) < 1e-6) | (np.abs(p[:, 1] - gy1) < 1e-6)
            p = p[~on]
        pts[i] = p
        kd[i] = cKDTree(p)
    SG, gtree = _ground_segments(G)
    _FORM.clear()
    _FORM.update(pts=pts, kd=kd, near=near, width=width, area=area, R=R, k=k, step=step_um, eps_r=eps_r,
                 every=max(1, int(round(screen_um / step_um))), SG=SG, gtree=gtree)
    rows = [i for i in need if i > 0]
    if workers > 1 and len(rows) > 1:
        with ProcessPoolExecutor(min(workers, len(rows)), mp_context=get_context("fork")) as ex:
            res = list(ex.map(_formula_row, rows, chunksize=max(1, len(rows) // (4 * workers))))
    else:
        res = [_formula_row(i) for i in rows]
    _FORM.clear()
    I = np.zeros((N, N))
    for i, row in res:
        I[i] = row
    C = np.zeros((N, N))
    iu = np.nonzero(np.triu(near, 1))
    gnd = iu[0] == 0
    C[iu] = np.where(gnd, I[iu[1], iu[0]], 0.5 * (I[iu] + I[iu[1], iu[0]]))
    fa, fb = np.nonzero(np.triu(want & ~near & ~near.T, 1))
    if len(fa):
        bb = shapely.bounds(np.array(metals, dtype=object))
        dx = np.maximum(0.0, np.maximum(bb[fa, 0] - bb[fb, 2], bb[fb, 0] - bb[fa, 2]))
        dy = np.maximum(0.0, np.maximum(bb[fa, 1] - bb[fb, 3], bb[fb, 1] - bb[fa, 3]))
        dd = np.maximum(np.hypot(dx, dy), 0.05)
        fG = G.area / max(shapely.box(*G.bounds).area, 1e-12)
        w1, w2 = width[fa], width[fb]
        m = dd * (dd + w1 + w2) / ((dd + w1) * (dd + w2))
        S = np.exp(-fG * dd / (dd + np.minimum(w1, w2)))
        C[fa, fb] = k * 0.5 * (perim[fa] + perim[fb]) * _kratio(m) * S
    C = -(C + C.T)
    info = {"n_sampled_pairs": int(len(iu[0])), "n_bbox_pairs": int(len(fa)), "min_gap_um": gap, "step_um": step_um,
            "samples": int(sum(len(p) for p in pts.values()))}
    if pairs is None:
        np.fill_diagonal(C, np.abs(C).sum(1) / (1 - ROW_SUM_EPS))
    info["total_s"] = time.perf_counter() - t0
    return C, info
