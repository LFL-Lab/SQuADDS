"""Physics-agnostic geometry blocks: node (per conductor), edge (per conductor pair) and global features, all in
physical units (log µm, ratios) so that nothing is fitted to a dataset."""

from __future__ import annotations

import hashlib
import os
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import shapely
from scipy.special import ellipk
from scipy.spatial import cKDTree

from .. import geometry as _geometry
from ..embed.proxy import _nthreads
from ..geometry import fourier_descriptors, sample_boundary, shape_metrics
from .scene import Scene

GRAN_R_UM = np.geomspace(0.25, 100.0, 16)
SEP_EDGES_UM = np.geomspace(0.5, 2000.0, 11)
MOMENTS = (0.5, 1.0, 2.0)
STEP_UM = 0.25


def _log(x):
    return float(np.log(max(float(x), 1e-9)))


def region(scene: Scene, i: int):
    """The shape that carries conductor i's edge: the etched pocket for the ground net, the metal otherwise."""
    return scene.pocket if scene.conductors[i].role == "ground" else scene.conductors[i].geom


def granulometry(g) -> np.ndarray:
    a0 = g.area
    return np.array([g.buffer(-r, quad_segs=4).area / a0 if a0 > 0 else 0.0 for r in GRAN_R_UM])


def node_features(scene: Scene, i: int) -> dict:
    g = region(scene, i)
    m = shape_metrics(g)
    f = {"role_ground": float(scene.conductors[i].role == "ground"),
         "log_area": _log(m["area"]), "log_perimeter": _log(m["perimeter"]), "compactness": m["compactness"],
         "solidity": m["solidity"], "log_min_halfwidth": _log(m["inscribed_radius"]),
         "log_aspect": abs(_log(m["bbox_w"]) - _log(m["bbox_h"])), "log_elongation": _log(m["elongation"]),
         "log_Imajor_over_A2": _log(m["I_major"] / max(m["area"], 1e-9) ** 2), "n_holes": float(m["n_holes"]),
         "log_width": _log(2 * m["area"] / m["perimeter"])}
    for j, v in enumerate(fourier_descriptors(g)):
        f[f"fourier_{j:02d}"] = float(np.log1p(50 * v))
    for j, v in enumerate(granulometry(g)):
        f[f"gran_{j:02d}"] = float(v)
    return f


def _kratio(m):
    m = np.clip(m, 1e-12, 1 - 1e-12)
    return ellipk(1 - m) / ellipk(m)


TOKEN_VARIANTS = {"exact": None, "td16": dict(d_far=100.0, k=16), "td64": dict(d_far=100.0, k=64), "tc600": dict(crop=600.0)}
TOKEN_VARIANT: dict = {"name": "exact"}  # PRIOR_SCHEME.md 14.3; "exact" = the published tokens


def set_token_variant(name: str) -> None:
    from ..geometry import MIC_TOL

    assert name in TOKEN_VARIANTS, name
    TOKEN_VARIANT["name"] = name
    MIC_TOL[0] = 0.01 if name == "exact" else 0.1


def _block_distances(pts: np.ndarray, tree) -> tuple:
    """Distances from the samples of a to b (k-d tree) for the active token variant, with weights (multiples of the
    sample step): the samples are grouped into consecutive blocks of k; a block whose first sample is farther than d_far
    from b is represented by that sample with weight = block size, the other blocks sample by sample. crop: blocks whose
    first sample is farther than the crop distance are dropped (their representatives kept if every block is dropped)."""
    v = TOKEN_VARIANTS[TOKEN_VARIANT["name"]]
    k = v.get("k", 64)
    r = v.get("crop", v.get("d_far"))
    rep = np.arange(0, len(pts), k)
    d_rep, _ = tree.query(pts[rep], workers=_nthreads())
    size = np.diff(np.r_[rep, len(pts)])
    far = d_rep > r
    near_pts = np.repeat(~far, size)
    d_near, _ = tree.query(pts[near_pts], workers=_nthreads())
    if "crop" in v:
        if not near_pts.any():
            return d_rep, size.astype(float)
        return d_near, np.ones(len(d_near))
    return np.r_[d_rep[far], d_near], np.r_[size[far].astype(float), np.ones(len(d_near))]


def edge_features(scene: Scene, i: int, j: int, bpts: dict, widths: dict) -> dict:
    """Symmetric pair features: separation spectrum (both directions averaged), minimum gap, coupling moments and the
    coplanar-strip conformal-map coupling (Gevorgian modulus; the ground net is treated as infinitely wide)."""
    f = {"is_ground_pair": float("ground" in (scene.conductors[i].role, scene.conductors[j].role))}
    hist = np.zeros(len(SEP_EDGES_UM) - 1)
    mom = np.zeros(len(MOMENTS))
    cps = 0.0
    dmin = np.inf
    for a, b in ((i, j), (j, i)):
        if TOKEN_VARIANT["name"] != "exact":  # approximate tokens (PRIOR_SCHEME.md 14.3)
            d, wq = _block_distances(bpts[a], cKDTree(bpts[b]))
            dmin = min(dmin, d.min())
            hist += 0.5 * np.histogram(d, bins=SEP_EDGES_UM, weights=STEP_UM * wq)[0]
            dd = np.maximum(d, 0.05)
            mom += 0.5 * np.array([np.sum(STEP_UM * wq * dd ** (-p)) for p in MOMENTS])
            w1, w2 = widths[a], widths[b]
            if np.isinf(w1) or np.isinf(w2):
                wf = w2 if np.isinf(w1) else w1
                mm = dd / (dd + wf)
            else:
                mm = dd * (dd + w1 + w2) / ((dd + w1) * (dd + w2))
            cps += 0.5 * np.sum(STEP_UM * wq * _kratio(mm))
            continue
        d, _ = cKDTree(bpts[b]).query(bpts[a], workers=_nthreads())
        dmin = min(dmin, d.min())
        hist += 0.5 * np.histogram(d, bins=SEP_EDGES_UM, weights=np.full(len(d), STEP_UM))[0]
        dd = np.maximum(d, 0.05)
        mom += 0.5 * np.array([np.sum(STEP_UM * dd ** (-p)) for p in MOMENTS])
        w1, w2 = widths[a], widths[b]
        if np.isinf(w1) or np.isinf(w2):
            wf = w2 if np.isinf(w1) else w1
            mm = dd / (dd + wf)
        else:
            mm = dd * (dd + w1 + w2) / ((dd + w1) * (dd + w2))
        cps += 0.5 * np.sum(STEP_UM * _kratio(mm))
    f["log_dmin"] = _log(dmin)
    for k, v in enumerate(hist):
        f[f"sep_{k:02d}"] = float(np.log1p(v))
    for p, v in zip(MOMENTS, mom):
        f[f"mom_{p:g}"] = _log(v)
    f["log_cps"] = _log(cps)
    f["_cps"] = cps  # raw, used for shielding shares
    return f


TREE_LEAFSIZE = 32  # nearest distances do not depend on the leaf size (cKDTree's own distance arithmetic); 32 is fastest
NN_KERNEL: dict = {"on": os.environ.get("TLSQ_NN_KERNEL", "1") == "1"}  # numba exact NN (nnk.py), probe-verified


def _tree(P: np.ndarray):
    """Nearest-sample structure of one conductor: the numba kernel when its probe reproduces cKDTree here, else cKDTree."""
    if NN_KERNEL["on"]:
        try:
            from . import nnk
        except ImportError:
            nnk = None
        if nnk is not None and nnk.available():
            return nnk.Tree(P)
    return cKDTree(P, leafsize=TREE_LEAFSIZE)


def _dist(tree, X: np.ndarray) -> np.ndarray:
    return tree.query(X)[0] if isinstance(tree, cKDTree) else tree.dist(X)


_EDGES_HI = np.r_[SEP_EDGES_UM[:-1], np.nextafter(SEP_EDGES_UM[-1], np.inf)]  # np.histogram: last bin closed
_CHUNK = 1 << 20  # distances per task (rows x samples)


def _dir_stats(X: np.ndarray, trees: list, bs: list, w1: float, widths: dict) -> tuple:
    """edge_features' per-direction statistics of the samples X of one conductor against each conductor in bs (all signal,
    or the ground alone): minimum distance, histogram counts, the three moment sums and the conformal sum. The same
    element-wise operations and pairwise row sums as edge_features (bit-identical)."""
    nb = len(bs)
    D = np.empty((nb, len(X)))
    for r, b in enumerate(bs):
        D[r] = _dist(trees[b], X)
    dmin = D.min(1)
    k = np.searchsorted(_EDGES_HI, D, side="right")
    k += (12 * np.arange(nb))[:, None]
    cnt = np.bincount(k.ravel(), minlength=12 * nb).reshape(nb, 12)[:, 1:11]
    del k
    dd = np.maximum(D, 0.05, out=D)
    mom = np.stack([np.sum(STEP_UM * dd ** (-p), axis=1) for p in MOMENTS], 1)
    w2 = np.array([widths[b] for b in bs])[:, None]
    if np.isinf(w1):
        mm = dd / (dd + w2)
    elif np.isinf(w2).any():
        mm = dd / (dd + w1)
    else:
        mm = dd * (dd + w1 + w2) / ((dd + w1) * (dd + w2))
    cps = np.sum(STEP_UM * _kratio(mm), axis=1)
    return dmin, cnt, mom, cps


# "bitwise" (default; training / reproduction): every feature bit-identical to the published code. "numex" (numerically exact;
# inference): node features and signal-signal directions computed once per class of translated copies (same shape to 1e-6
# um, same offset) and shared; values differ from bitwise in the last bits only (agents/embedspeed/RESULTS.md).
EMBED_MODE: dict = {"mode": os.environ.get("TLSQ_EMBED_MODE", "bitwise")}
FAR_DECIMATION: dict = {"k": 0, "d_far": 100.0}
# Edge storage: False (default; training) = a dict per pair; True (inference) = EdgeTable, one float64 array per key.
EDGE_TABLE: dict = {"on": os.environ.get("TLSQ_EDGE_TABLE", "0") == "1"}


class EdgeTable(Mapping):
    """Read-only complete-graph edges: pairs i < j in combinations order, one float64 column per key (same values as the
    dict per pair; `npcols` are returned as numpy float64, the others as Python floats). table[(i, j)] builds that pair's
    dict on request; matrix(keys) gives the (P, len(keys)) array without building any dict."""

    def __init__(self, n: int, cols: dict, npcols=()):
        self.n, self.cols, self.npcols = n, cols, frozenset(npcols)

    def _r(self, p) -> int:
        i, j = int(p[0]), int(p[1])
        if not 0 <= i < j < self.n:
            raise KeyError(p)
        return i * (2 * self.n - i - 1) // 2 + j - i - 1

    def __getitem__(self, p) -> dict:
        r = self._r(p)
        return {k: (v[r] if k in self.npcols else float(v[r])) for k, v in self.cols.items()}

    def __iter__(self):
        return ((i, j) for i in range(self.n) for j in range(i + 1, self.n))

    def __len__(self) -> int:
        return self.n * (self.n - 1) // 2

    def matrix(self, keys, dtype=np.float32) -> np.ndarray:
        if not len(self):
            return np.zeros((0, len(keys)), dtype)
        return np.stack([self.cols[k] for k in keys], 1).astype(dtype)

    def with_cols(self, new: dict, drop=(), npcols=()) -> "EdgeTable":
        cols = {k: v for k, v in self.cols.items() if k not in drop}
        return EdgeTable(self.n, {**cols, **new}, (self.npcols | frozenset(npcols)) - frozenset(drop))


def edge_matrix(edges, keys, n: int, dtype=np.float32) -> np.ndarray:
    """(P, len(keys)) array of the edge features (pairs i < j): the table's columns, or the per-pair dicts as before."""
    if isinstance(edges, EdgeTable):
        return edges.matrix(keys, dtype)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return np.array([[edges[p][k] for k in keys] for p in pairs], dtype).reshape(len(pairs), len(keys))  # approximation arm (agents/embedspeed/PROTOCOL.md Part A); k = 0: exact


def _dir_stats_far(X: np.ndarray, trees: list, bs: list, w1: float, widths: dict) -> tuple:
    """Approximate _dir_stats (block decimation of PRIOR_SCHEME.md 14.3.1): blocks of k consecutive samples of a whose first
    sample is farther than d_far from b are represented by that sample with weight k (the last block by its size). Distances
    per row (far representatives, then the near samples), statistics for the whole chunk at once (weighted sums)."""
    k, r = FAR_DECIMATION["k"], FAR_DECIMATION["d_far"]
    rep = np.arange(0, len(X), k)
    size = np.diff(np.r_[rep, len(X)])
    nb = len(bs)
    Xr = X[rep]
    ds, ws, rows = [], [], []
    for row, b in enumerate(bs):
        d_rep = _dist(trees[b], Xr)
        far = d_rep > r
        if far.all():
            d, wq = d_rep, size.astype(float)
        else:
            near = np.repeat(~far, size)
            d = np.r_[d_rep[far], _dist(trees[b], X[near])]
            wq = np.r_[size[far].astype(float), np.ones(int(near.sum()))]
        ds.append(d)
        ws.append(wq)
        rows.append(np.full(len(d), row))
    d, wq, rid = np.concatenate(ds), np.concatenate(ws), np.concatenate(rows)
    starts = np.r_[0, np.cumsum([len(x) for x in ds])[:-1]]
    dmin = np.minimum.reduceat(d, starts)
    kb = np.searchsorted(_EDGES_HI, d, side="right")
    cnt = np.rint(np.bincount(rid * 12 + kb, weights=wq, minlength=12 * nb)).astype(np.int64).reshape(nb, 12)[:, 1:11]
    dd = np.maximum(d, 0.05)
    mom = np.stack([np.bincount(rid, weights=STEP_UM * wq * dd ** (-p), minlength=nb) for p in MOMENTS], 1)
    w2 = np.array([widths[b] for b in bs])[rid]
    if np.isinf(w1):
        mm = dd / (dd + w2)
    elif np.isinf(w2).any():
        mm = dd / (dd + w1)
    else:
        mm = dd * (dd + w1 + w2) / ((dd + w1) * (dd + w2))
    cps = np.bincount(rid, weights=STEP_UM * wq * _kratio(mm), minlength=nb)
    return dmin, cnt, mom, cps


def _pmap(fn, items: list, threads: int) -> list:
    if threads < 2 or len(items) < 2:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads, thread_name_prefix="tlsq-edges") as ex:
        return list(ex.map(fn, items))


# Optional persistent cache (exact): per-body node features and per-direction pair statistics, keyed by the body's region
# (WKB) and role, under a key of this code, the parameters and the machine / library versions (a value is reused only where
# this machine would recompute it bit for bit). Off unless TLSQ_EMBED_CACHE (or EMBED_CACHE["dir"]) names a directory.
EMBED_CACHE: dict = {"dir": os.environ.get("TLSQ_EMBED_CACHE") or None}


def _code_key() -> str:
    import platform

    import scipy

    h = hashlib.sha256()
    for f in (__file__, _geometry.__file__):
        h.update(Path(f).read_bytes())
    h.update(repr((platform.machine(), platform.system(), np.__version__, scipy.__version__, shapely.__version__, STEP_UM,
                   SEP_EDGES_UM.tolist(), MOMENTS, TREE_LEAFSIZE, TOKEN_VARIANT["name"], EMBED_MODE["mode"],
                   NN_KERNEL["on"])).encode())
    return h.hexdigest()[:24]


def body_keys(scene: Scene) -> list:
    return [hashlib.sha256(c.role.encode() + shapely.to_wkb(region(scene, i))).hexdigest()[:32]
            for i, c in enumerate(scene.conductors)]


class _Cache:
    def __init__(self, root):
        self.d = Path(root) / _code_key()
        for sub in ("dir", "node"):
            (self.d / sub).mkdir(parents=True, exist_ok=True)

    def _put(self, path: Path, write):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            write(fh)
        os.replace(tmp, path)

    def nodes(self, keys: list) -> dict:
        out = {}
        for i, k in enumerate(keys):
            p = self.d / "node" / f"{k}.pkl"
            if p.exists():
                out[i] = pickle.loads(p.read_bytes())
        return out

    def put_node(self, key: str, f: dict):
        self._put(self.d / "node" / f"{key}.pkl", lambda fh: pickle.dump(f, fh, protocol=5))

    def rows(self, ka: str):
        p = self.d / "dir" / f"{ka}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return {str(k): r for r, k in enumerate(z["keys"])}, z["dmin"], z["cnt"], z["mom"], z["cps"]

    def put_rows(self, ka: str, keys: list, dmin, cnt, mom, cps):
        old = self.rows(ka)
        if old is not None:
            idx, *arr = old
            ks = set(keys)
            extra = [r for k, r in idx.items() if k not in ks]
            keys = list(keys) + [k for k, r in idx.items() if k not in ks]
            dmin, cnt, mom, cps = (np.concatenate([x, y[extra]]) for x, y in zip((dmin, cnt, mom, cps), arr))
        self._put(self.d / "dir" / f"{ka}.npz",
                  lambda fh: np.savez(fh, keys=np.array(keys), dmin=dmin, cnt=cnt, mom=mom, cps=cps))


    def trim(self):
        """Size bound (TLSQ_EMBED_CACHE_MB): the least recently written entries are removed down to 80 % of the bound."""
        mb = os.environ.get("TLSQ_EMBED_CACHE_MB")
        if not mb:
            return
        lim = float(mb) * 2**20
        files = [e for sub in ("dir", "node") for e in os.scandir(self.d / sub) if e.is_file()]
        tot = sum(e.stat().st_size for e in files)
        if tot <= lim:
            return
        for e in sorted(files, key=lambda e: e.stat().st_mtime):
            if tot <= 0.8 * lim:
                break
            try:
                tot -= e.stat().st_size
                os.remove(e.path)
            except OSError:
                pass


def _cache():
    return _Cache(EMBED_CACHE["dir"]) if EMBED_CACHE["dir"] else None


def shape_classes(scene: Scene, bpts: dict | None = None) -> tuple:
    """Translation classes of the signal bodies (numex mode): class id per conductor (-1 = ground, its own class) from the
    coordinates relative to the bounding-box corner rounded to 1e-6 um and the ring structure, and, with bpts, from the
    boundary samples relative to that corner rounded to 1e-9 um (a translated copy can get one sample more or fewer when a
    ring length is a multiple of the step); integer origins (1e-6 um)."""
    ids, cid, org = {}, np.full(scene.n, -1), np.zeros((scene.n, 2), np.int64)
    for i, c in enumerate(scene.conductors):
        if c.role == "ground":
            continue
        g = region(scene, i)
        o = np.array(g.bounds[:2])
        rings = [len(r.coords) for q in getattr(g, "geoms", [g]) for r in (q.exterior, *q.interiors)]
        h = hashlib.sha1(repr((c.role, g.geom_type, rings)).encode() + (np.round(shapely.get_coordinates(g) - o, 6) + 0.0).tobytes())
        if bpts is not None:
            h.update((np.round(bpts[i] - o, 9) + 0.0).tobytes())
        cid[i] = ids.setdefault(h.hexdigest(), len(ids))
        org[i] = np.rint(o * 1e6).astype(np.int64)
    return cid, org


def node_features_all(scene: Scene, keys: list | None = None) -> list:
    """[node_features(scene, i) for every conductor], threads over the conductors, reusing the persistent cache (numex:
    once per translation class)."""
    c = _cache()
    keys = keys if keys is not None or c is None else body_keys(scene)
    hit = c.nodes(keys) if c else {}
    miss = [i for i in range(scene.n) if i not in hit]
    rep = {}
    if EMBED_MODE["mode"] == "numex":
        cid = shape_classes(scene)[0]
        first = {}
        rep = {i: first.setdefault(cid[i], i) for i in miss if cid[i] >= 0}
        rep = {i: r for i, r in rep.items() if r != i}
    new = dict(zip([i for i in miss if i not in rep], _pmap(lambda i: node_features(scene, i), [i for i in miss if i not in rep], _nthreads())))
    new.update({i: dict(new[r]) for i, r in rep.items()})
    if c:
        for i, f in new.items():
            c.put_node(keys[i], f)
        c.trim()
    return [hit[i] if i in hit else new[i] for i in range(scene.n)]


def edge_features_all(scene: Scene, bpts: dict, widths: dict, keys: list | None = None) -> dict:
    """{(i, j): edge_features(scene, i, j, bpts, widths)} for every pair i < j, bit-identical, batched: one k-d tree per
    conductor, every direction a -> b as rows of one distance array per (a, chunk of b), threads over the chunks (the
    queries and ufuncs release the GIL); directions found in the persistent cache are not recomputed."""
    n = scene.n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if TOKEN_VARIANT["name"] != "exact" or n < 2:
        return {(i, j): edge_features(scene, i, j, bpts, widths) for i, j in pairs}
    th = _nthreads()
    DM, CP = np.empty((n, n)), np.empty((n, n))
    CN, MO = np.empty((n, n, 10), np.int64), np.empty((n, n, len(MOMENTS)))
    c = _cache() if not FAR_DECIMATION["k"] else None
    if c and keys is None:
        keys = body_keys(scene)
    todo = {a: [b for b in range(n) if b != a] for a in range(n)}
    if c:
        for a in range(n):
            got = c.rows(keys[a])
            if got is None:
                continue
            idx, dm, cn, mo, cp = got
            have = [b for b in todo[a] if keys[b] in idx]
            r = [idx[keys[b]] for b in have]
            DM[a, have], CN[a, have], MO[a, have], CP[a, have] = dm[r], cn[r], mo[r], cp[r]
            todo[a] = [b for b in todo[a] if keys[b] not in idx]
    share = None
    if EMBED_MODE["mode"] == "numex":
        cid, org = shape_classes(scene, bpts)
        A, B = np.nonzero((cid[:, None] >= 0) & (cid[None, :] >= 0) & ~np.eye(n, dtype=bool))
        K = np.c_[cid[A], cid[B], org[B] - org[A]]
        _, first, inv = np.unique(K, axis=0, return_index=True, return_inverse=True)
        R = first[inv.ravel()]
        share = (A, B, A[R], B[R])
        drop = np.zeros((n, n), bool)
        drop[A[R != np.arange(len(A))], B[R != np.arange(len(A))]] = True
        todo = {a: [b for b in bs if not drop[a, b]] for a, bs in todo.items()}
    tasks = []
    for a, bs in todo.items():
        sig = [b for b in bs if not np.isinf(widths[b])]
        tasks += [(a, [b]) for b in bs if np.isinf(widths[b])]
        step = max(1, _CHUNK // len(bpts[a]))
        tasks += [(a, sig[k:k + step]) for k in range(0, len(sig), step)]
    tasks.sort(key=lambda t: -len(t[1]) * len(bpts[t[0]]))
    need = sorted({b for _, bs in tasks for b in bs})
    trees = dict(zip(need, _pmap(lambda i: _tree(bpts[i]), need, th)))

    def run(t):
        a, bs = t
        f = _dir_stats_far if FAR_DECIMATION["k"] else _dir_stats
        DM[a, bs], CN[a, bs], MO[a, bs], CP[a, bs] = f(bpts[a], trees, bs, widths[a], widths)

    _pmap(run, tasks, th)
    if share is not None:
        A, B, RA, RB = share
        DM[A, B], CN[A, B], MO[A, B], CP[A, B] = DM[RA, RB], CN[RA, RB], MO[RA, RB], CP[RA, RB]
    if c:
        for a in {t[0] for t in tasks}:
            bs = [b for b in range(n) if b != a]
            c.put_rows(keys[a], [keys[b] for b in bs], DM[a, bs], CN[a, bs], MO[a, bs], CP[a, bs])
        c.trim()
    return _assemble(scene, DM, CN, MO, CP)


def _assemble(scene: Scene, DM, CN, MO, CP) -> dict:
    """Pair dicts from the per-direction statistics, in edge_features' operation order and key order."""
    n = scene.n
    I, J = np.triu_indices(n, 1)
    dmin = np.minimum(DM[I, J], DM[J, I])
    hist = 0.5 * (STEP_UM * CN[I, J]) + 0.5 * (STEP_UM * CN[J, I])
    mom = 0.5 * MO[I, J] + 0.5 * MO[J, I]
    cps = 0.5 * CP[I, J] + 0.5 * CP[J, I]
    gnd = np.array([c.role == "ground" for c in scene.conductors])
    cols = {"is_ground_pair": (gnd[I] | gnd[J]).astype(float), "log_dmin": np.log(np.maximum(dmin, 1e-9))}
    sep = np.log1p(hist)
    for k in range(sep.shape[1]):
        cols[f"sep_{k:02d}"] = sep[:, k]
    lm = np.log(np.maximum(mom, 1e-9))
    for k, p in enumerate(MOMENTS):
        cols[f"mom_{p:g}"] = lm[:, k]
    cols["log_cps"] = np.log(np.maximum(cps, 1e-9))
    if EDGE_TABLE["on"]:
        return EdgeTable(n, {**cols, "_cps": cps}, ("_cps",))
    keys = list(cols)
    rows = zip(*[cols[k].tolist() for k in keys], list(cps))
    return {(i, j): {**dict(zip(keys, r[:-1])), "_cps": r[-1]} for i, j, r in zip(I.tolist(), J.tolist(), rows)}


def global_features(scene: Scene) -> dict:
    st = scene.stack
    n_sig = sum(c.role == "signal" for c in scene.conductors)
    return {"n_metal_layers": float(st["n_metal_layers"]), "log_metal_t": _log(st["metal_thickness_um"]),
            "log_substrate_t": _log(st["substrate_thickness_um"]), "eps_eff": (st["substrate_eps_r"] + 1) / 2,
            "n_signal": float(n_sig), "log_chip_area": _log(scene.chip.area),
            "log_pocket_area": _log(scene.pocket.area)}


def boundary_points(scene: Scene) -> tuple[dict, dict]:
    def one(i):
        g = region(scene, i)
        return sample_boundary(g, STEP_UM), np.inf if scene.conductors[i].role == "ground" else 2 * g.area / g.length

    out = _pmap(one, list(range(scene.n)), _nthreads())
    return {i: p for i, (p, _) in enumerate(out)}, {i: w for i, (_, w) in enumerate(out)}
