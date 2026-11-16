"""Local-gap (graded) lite micro-MoM prior: PRIOR_SCHEME.md Amendment MESH (2026-10-11).

The lite prior meshes every metal edge at h = gap / ppg with gap the layout's global smallest conductor gap. A sliver
(exp12: the 35 nm end slit of a pad lead, 10.2 um long) therefore sets the whole mesh, and h sits on the size / 4096
floor at every ppg (the resolution knob is dead). Here the panel size is set by the LOCAL gap of each quadtree cell:

    h(cell) = clamp(local_gap(cell) / ppg, h_floor, H),   H = G_ref / ppg,   h_floor = size / FLOOR_DIV

local_gap(cell) = min over the metals k the cell touches and the other metals j of distance(cell & metal_k, metal_j)
(the gap of the edge stretch inside the cell, never smaller than the global smallest gap); G_ref = the smallest pair
gap among conductor pairs whose gap is >= SLIVER_UM ("sliver" pairs only refine their own neighbourhood). Same lattice
(H 2^-k, anchored to the signal bounding box), exact cut panels, strip merge bounded by each panel's own size, slab
kernel and exact near field as `fastprior.micro_mom`. A layout without a sliver pair is passed to `fastprior.micro_mom`
unchanged (bit-identical prior). Resolution descriptor: proxy_min_gap_um = G_ref, proxy_h_min_um = H (log ppg).
"""

from __future__ import annotations

import time

import numpy as np
import shapely

from . import fastprior as F
from . import proxy as P

SLIVER_UM = 0.25  # pair gaps below this are slivers (no labelled layout has a gap in [0.04, 1.99] um; exp12 slits 34.9-35.6 nm)
FLOOR_DIV = 4096  # h_floor = layout size / FLOOR_DIV (the existing floor)


def ref_gap(metals: list, sliver: float = SLIVER_UM) -> tuple:
    """(G_ref, smallest gap, layout size, number of sliver pairs). G_ref = smallest pair gap >= sliver; when every pair
    is a sliver, the edge-length-weighted 5th percentile of the local gap of the boundary (0.25 um samples)."""
    gmin, size = F._gap_size(metals)
    n = len(metals)
    d = np.array([shapely.distance(metals[i], metals[j]) for i in range(n) for j in range(i + 1, n)])
    ok = d >= sliver
    if ok.any():
        return float(d[ok].min()), float(gmin), float(size), int((~ok).sum())
    lg = []
    for i, g in enumerate(metals):
        others = shapely.union_all([metals[k] for k in range(n) if k != i])
        for ln in getattr(g.boundary, "geoms", [g.boundary]):
            pts = shapely.line_interpolate_point(ln, np.arange(0, ln.length, 0.25))
            lg.append(shapely.distance(pts, others))
    return float(max(np.percentile(np.concatenate(lg), 5), gmin)), float(gmin), float(size), int((~ok).sum())


def _local_gap(metals: list, boxes, hit, inn, pcs, sel) -> np.ndarray:
    """local_gap of the cells sel (indices): min over touched metals k and other metals j of distance(cell & k, j)."""
    lg = np.full(len(sel), np.inf)
    m = len(metals)
    for k in range(m):
        t = hit[k, sel]
        if not t.any():
            continue
        ix = sel[t]
        geo = np.where(inn[k, ix], boxes[ix], pcs[k][ix])
        bad = np.array([g is None for g in geo])
        geo[bad] = boxes[ix][bad]
        dk = np.full(len(ix), np.inf)
        for j in range(m):
            if j != k:
                dk = np.minimum(dk, shapely.distance(geo, metals[j]))
        lg[t] = np.minimum(lg[t], dk)
    return lg


def anchored_cells_local(metals: list, H: float, h_floor: float, ppg: float, grade: float = 2.0, majority: bool = False):
    """`proxy.anchored_cells` on the lattice of H (anchored to the signal bounding box) with the finest level of every
    cell set by its local gap: a cell (cut or whole) is final once h <= clamp(local_gap / ppg, h_floor, H). Returns
    centres (n, 2), sizes (n, 2), owner (n,)."""
    edges = shapely.union_all([g.boundary for g in metals])
    sig = metals[1:] if len(metals) > 1 else metals
    ax, ay = shapely.union_all(sig).bounds[:2]
    X0, Y0, X1, Y1 = shapely.union_all(metals).bounds
    span = max(X1 - X0, Y1 - Y0)
    K = max(0, int(np.ceil(np.log2(max(span / 16, H) / H))))
    h0 = H * 2**K
    ix0, iy0 = int(np.floor((X0 - ax) / h0)), int(np.floor((Y0 - ay) / h0))
    ix1, iy1 = int(np.ceil((X1 - ax) / h0)), int(np.ceil((Y1 - ay) / h0))
    gx, gy = np.meshgrid(ax + h0 * np.arange(ix0, ix1), ay + h0 * np.arange(iy0, iy1))
    xs, ys, h = gx.ravel(), gy.ravel(), np.full(gx.size, h0)
    gq = P._Geoms(list(metals) + [edges])
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
        cand = h <= H * 1.0001
        last = cand & (h <= h_floor * 1.0001)
        need = np.nonzero(cand & ~last)[0]
        if len(need):
            tgt = np.clip(_local_gap(metals, boxes, hit, inn, pcs, need) / ppg, h_floor, H)
            last[need] = h[need] <= tgt * 1.0001
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


def mesh_local(metals: list, H: float, h_floor: float, ppg: float, majority: bool = False) -> tuple:
    """Strip-merged local mesh (strip length <= 16 x the panel's own size and <= the distance to the nearest vertex).
    Returns (centres, sizes, owners, raw count)."""
    raw = anchored_cells_local(metals, H, h_floor, ppg, majority=majority)
    c, wh, o = F.strip_merge(*raw, metals, h_floor)
    return c, wh, o, len(raw[0])


def _solve(metals: list, H: float, h_floor: float, ppg: float, eps_r: float) -> tuple:
    """`fastprior._window_solve(all_rhs=True)` on the local mesh: rejected solves (not PD, a metal without a panel, a
    non-negative mutual entry) refine (H, h_floor) / 2 up to LITE_REFINE times, then majority-owned cut cells."""
    t_mesh = t_sol = 0.0
    cols = np.arange(len(metals))
    refined, majority = 0, False
    while True:
        t = time.perf_counter()
        c, wh, o, n_raw = mesh_local(metals, H, h_floor, ppg, majority)
        t_mesh += time.perf_counter() - t
        t = time.perf_counter()
        try:
            if len(c) <= P.DENSE_MAX:
                A = P._assemble(c, wh, P.rect_self(wh[:, 0], wh[:, 1]), eps_r, P.rect_potential)
                Q = P._chol_solve(A, (o[:, None] == cols[None, :]).astype(float))
                del A
            else:
                Q = P.mom_solve_fast(c, wh, o, len(metals), eps_r)
            R = np.stack([Q[o == a].sum(0) for a in range(len(metals))], 1)
            off = ~np.eye(len(metals), dtype=bool)
            ok = bool(np.all(np.bincount(o, minlength=len(metals)) > 0) and np.all(R[off] < 0))
        except np.linalg.LinAlgError:
            ok = False
        t_sol += time.perf_counter() - t
        if ok or majority:
            break
        if refined < F.LITE_REFINE:
            H, h_floor, refined = H / 2, h_floor / 2, refined + 1
        else:
            majority = True
    return R, {"H_um": H, "h_floor_um": h_floor, "h_min_um": float(wh.min()), "panels": len(c), "raw": n_raw,
               "refined": refined, "majority": majority, "mesh_s": t_mesh, "solve_s": t_sol}


def micro_mom_loc(G, conds: list, eps_r: float = 11.45, ppg: float = 3.0, sliver: float = SLIVER_UM,
                  floor_div: float = FLOOR_DIV, canonical: bool = True) -> tuple:
    """Local-gap lite prior of a <= 8-conductor layout: (C (N+1, N+1) fF, far_mask (all False), info). Layouts without a
    sliver pair: `fastprior.micro_mom(budget=inf, cells_per_gap=ppg)` (the existing liteppg prior, bit-identical).
    Mean over the canonical frames of symmetric layouts, returned in the input order (as micro_mom)."""
    t_all = time.perf_counter()
    metals = [G] + list(conds)
    gref, gmin, size, n_sl = ref_gap(metals, sliver)
    if n_sl == 0 or len(conds) > 8:  # no sliver (or windows: out of scope) -> the existing prior
        C, far, info = F.micro_mom(G, conds, eps_r, budget=np.inf, cells_per_gap=ppg)
        info.update(local=False, G_ref_um=float(info["min_gap_um"]), H_um=float(min(info["h_um"])), n_sliver_pairs=n_sl)
        return C, far, info
    runs = []
    frames = F.canonical_frames(G, conds) if canonical else [(G, list(conds), np.arange(len(conds)))]
    for Gc, cc, order in frames:
        R, wi = _solve([Gc] + list(cc), gref / ppg, size / floor_div, ppg, eps_r)
        C = 0.5 * (R + R.T)
        inv = np.r_[0, np.argsort(order) + 1]
        runs.append((C[np.ix_(inv, inv)], wi))
    C = sum(r[0] for r in runs) / len(runs)
    wi = runs[0][1]
    info = {"local": True, "n_windows": 1, "n_frames": len(runs), "min_gap_um": gref, "G_ref_um": gref, "gmin_um": gmin,
            "n_sliver_pairs": n_sl, "H_um": wi["H_um"], "h_floor_um": wi["h_floor_um"], "h_um": [wi["H_um"]],
            "h_min_um": wi["h_min_um"], "panels": [wi["panels"]], "panels_per_gap": [gref / wi["H_um"]],
            "majority": [wi["majority"]], "refined": [wi["refined"]],
            "mesh_s": sum(r[1]["mesh_s"] for r in runs), "solve_s": sum(r[1]["solve_s"] for r in runs)}
    info["total_s"] = time.perf_counter() - t_all
    return C, np.zeros(C.shape, bool), info
