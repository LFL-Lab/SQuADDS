"""Local, window-invariant context tokens (PRIOR_SCHEME.md 14.5 E4; replace the chip / pocket areas): for every signal
conductor and radius R, the fractions of its surrounding annulus (conductor buffered by R, minus the conductor) that are
ground metal, other-conductor metal and etched chip; the rest of the annulus lies outside the chip (no metal). Pair
tokens = mean over the pair's signal ends (a ground pair: its signal end). Identical copy: tlsq_infer/localtok.py."""

from __future__ import annotations

import numpy as np
import shapely

from ..geometry import sample_boundary
from ..embed.proxy import _nthreads
from .blocks import MOMENTS, SEP_EDGES_UM, STEP_UM, EdgeTable, _dist, _kratio, _pmap, _tree

RADII_UM = (50.0, 150.0, 400.0)
KEYS = [f"lt_{k}{int(r)}" for r in RADII_UM for k in ("g", "o", "e")]


def node_tokens(scene) -> np.ndarray:
    """(n, len(KEYS)) per conductor (ground row = NaN)."""
    conds = scene.conductors
    n = len(conds)
    G = next(c.geom for c in conds if c.role == "ground")
    sig = [k for k, c in enumerate(conds) if c.role != "ground"]
    tree = shapely.STRtree([conds[k].geom for k in sig])
    out = np.full((n, len(KEYS)), np.nan)

    def one(k):
        g = conds[k].geom
        vals = []
        for R in RADII_UM:
            ring = g.buffer(R, quad_segs=4).difference(g)
            A = ring.area
            box = ring.bounds
            gm = shapely.intersection(shapely.clip_by_rect(G, *box), ring).area
            near = [sig[i] for i in tree.query(ring, predicate="intersects") if sig[i] != k]
            om = shapely.intersection(shapely.union_all([conds[i].geom for i in near]), ring).area if near else 0.0
            inchip = shapely.intersection(scene.chip, ring).area
            vals += [gm / A, om / A, max(inchip - gm - om, 0.0) / A]
        return vals

    for k, vals in zip(sig, _pmap(one, sig, _nthreads())):  # threads: shapely releases the GIL
        out[k] = vals
    return out


LG_R_UM = 100.0
TOK = ["log_dmin"] + [f"sep_{k:02d}" for k in range(10)] + ["mom_0.5", "mom_1", "mom_2", "log_cps"]
LG_KEYS = [f"lg_{k}" for k in TOK]


def ground_tokens(scene) -> np.ndarray:
    """(n, len(LG_KEYS)) per signal conductor: the ground-pair tokens of blocks.edge_features (separation spectrum, minimum
    gap, coupling moments, coplanar-strip coupling) from the ground-edge samples within LG_R_UM of the conductor only
    (fixed radius: independent of the ground window)."""
    conds = scene.conductors
    out = np.full((len(conds), len(LG_KEYS)), np.nan)

    def one(k):
        g = conds[k].geom
        b = g.bounds
        P = shapely.clip_by_rect(scene.pocket, b[0] - LG_R_UM, b[1] - LG_R_UM, b[2] + LG_R_UM, b[3] + LG_R_UM)
        pj = sample_boundary(g, STEP_UM)
        pg = sample_boundary(P, STEP_UM) if not P.is_empty else np.zeros((0, 2))
        if len(pg):
            pg = pg[_dist(_tree(pj), pg) < LG_R_UM]  # drops the clip-box edges (>= LG_R_UM away) and the far ground
        if not len(pg):
            return None
        wj = 2 * g.area / g.length
        hist = np.zeros(len(SEP_EDGES_UM) - 1)
        mom = np.zeros(len(MOMENTS))
        cps, dmin = 0.0, np.inf
        for a_, b_ in ((pj, pg), (pg, pj)):
            d = _dist(_tree(b_), a_)
            dmin = min(dmin, d.min())
            hist += 0.5 * np.histogram(d, bins=SEP_EDGES_UM, weights=np.full(len(d), STEP_UM))[0]
            dd = np.maximum(d, 0.05)
            mom += 0.5 * np.array([np.sum(STEP_UM * dd ** (-p)) for p in MOMENTS])
            cps += 0.5 * np.sum(STEP_UM * _kratio(dd / (dd + wj)))
        return [np.log(max(dmin, 1e-9))] + list(np.log1p(hist)) + [np.log(max(v, 1e-9)) for v in mom] + [np.log(max(cps, 1e-9))]

    sig = [k for k, c in enumerate(conds) if c.role != "ground"]
    for k, v in zip(sig, _pmap(one, sig, _nthreads())):  # threads: shapely and the k-d tree queries release the GIL
        if v is not None:
            out[k] = v
    return out


def add_tokens(scene, row: dict) -> dict:
    """Row (scene graph dict) with the KEYS added to every edge (shallow copy)."""
    return add_from_tokens(row, node_tokens(scene), ground_tokens(scene), [c.role != "ground" for c in scene.conductors])


def add_from_tokens(row: dict, T: np.ndarray, Lg: np.ndarray, sig) -> dict:
    """Edges + KEYS (mean over the signal ends) + LG_KEYS (ground pairs: the fixed-radius tokens of the signal end; other
    pairs: their own tokens, unchanged)."""
    sig = np.asarray(sig)
    if isinstance(row["edges"], EdgeTable):
        return {**row, "edges": _add_table(row["edges"], T, Lg, sig)}
    edges = {}
    for (i, j), f in row["edges"].items():
        ends = [e for e in (i, j) if sig[e]]
        new = {k: float(x) for k, x in zip(KEYS, T[ends].mean(0))}
        if len(ends) == 1 and np.all(np.isfinite(Lg[ends[0]])):
            new.update({k: float(x) for k, x in zip(LG_KEYS, Lg[ends[0]])})
        else:
            new.update({f"lg_{k}": float(f[k]) for k in TOK})
        edges[(i, j)] = {**f, **new}
    return {**row, "edges": edges}


def _add_table(E: EdgeTable, T: np.ndarray, Lg: np.ndarray, sig: np.ndarray) -> EdgeTable:
    """add_from_tokens on an EdgeTable, per column with the same arithmetic: the mean over two signal ends is (a + b) / 2
    (numpy's mean of two rows), over one end the row itself."""
    I, J = np.triu_indices(E.n, 1)
    sI, sJ = sig[I], sig[J]
    both, oi, oj = sI & sJ, sI & ~sJ, ~sI & sJ
    M = np.full((len(I), len(KEYS)), np.nan)
    M[both] = (T[I[both]] + T[J[both]]) / 2
    M[oi], M[oj] = T[I[oi]], T[J[oj]]
    end = np.where(oi, I, J)
    use = (oi | oj) & np.isfinite(Lg[end]).all(1)
    new = {k: M[:, c] for c, k in enumerate(KEYS)}
    for c, k in enumerate(TOK):
        new[f"lg_{k}"] = np.where(use, Lg[end][:, c], E.cols[k])
    return E.with_cols(new)
