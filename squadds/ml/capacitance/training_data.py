"""Training data of the planar-capacitance networks: the P2d shard layout, label reading and QC, and shard builders.

A shard directory holds one ``n0NN.npz`` per conductor count N (ground included) and ``schema.json``. Arrays per file
(L layouts, P = N(N-1)/2 pairs ``i < j``):

    key (L,) str   split (L,) str   family (L,) str
    X (L, N, 61) f32   node vector (60 features + missing_log_min_halfwidth), node 0 = ground G
    E (L, P, 42) f32   pair vector          G (L, 4) f32   global vector
    C (L, N, N) f32    Maxwell matrix in fF, ground first (NaN = unknown entry, masked in training)
    sig (L, N) bool    signal mask          pri (L, P, 2) f32   positive prior mutuals (fF) at 1.5 and 3 panels per gap
    pairs (P, 2) int

``build_shards`` turns layouts (GDS) and their labels (a Maxwell matrix with net names, or a Q3D ``cap_matrix.csv``) into
shards with the same embedding and the ``optimized_physics_prior`` solves the released models read. Net names are matched by
name and never merged; every label passes QC (symmetry, signs, closure, positive definiteness) or is refused.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GROUND = "G"
PRIOR_LEVELS = (1.5, 3.0)
TO_FF = {"F": 1e15, "nF": 1e6, "pF": 1e3, "fF": 1.0, "aF": 1e-3}
ARRAYS = ("key", "split", "family", "X", "E", "G", "C", "sig", "pri", "pairs")


class LabelQCError(ValueError):
    """A label matrix failed QC (symmetry, Maxwell signs, closure, positive definiteness, net names)."""


def base_schema() -> dict:
    """Feature names and order of the networks' input (nodekeys 61, edgekeys 42, globkeys 4) and the prior's physics."""
    from ._core import paper as PA
    from .models import MANIFEST

    s = json.loads((Path(__file__).with_name("_core") / "schema_p2d.json").read_text(encoding="utf-8"))
    st = MANIFEST["prior_stacks"]["corrected"]
    return {
        "nodekeys": list(s["nodekeys"]),
        "edgekeys": list(s["edgekeys"]),
        "globkeys": list(s["globkeys"]),
        "physics": {
            "prior_method": "local_gap",
            "alpha": PA.ALPHA,
            "ensemble": "arithmetic mean of complete predicted Maxwell matrices",
            "prior_version": st["version"],
            "prior_stack": "corrected",
            "levels_ppg": list(PRIOR_LEVELS),
        },
    }


def pairs_of(n: int) -> np.ndarray:
    return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64).reshape(-1, 2)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- shards
def read_schema(shard_dir) -> dict:
    return json.loads((Path(shard_dir) / "schema.json").read_text(encoding="utf-8"))


def shard_files(shard_dir) -> list[Path]:
    files = sorted(Path(shard_dir).glob("n*.npz"))
    if not files:
        raise FileNotFoundError(f"no n*.npz shards in {shard_dir}")
    return files


def read_shards(shard_dir) -> list[dict]:
    """Every shard file as a dict of arrays (sorted by N), plus ``n`` and ``file``."""
    out = []
    for p in shard_files(shard_dir):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["n"], d["file"] = int(d["sig"].shape[1]), p.name
        out.append(d)
    return out


def read_names(shard_dir) -> dict:
    """{key: net names (G first)} of a shard dir built here (``names.json``); {} for shards without it."""
    p = Path(shard_dir) / "names.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def shard_summary(shard_dir) -> dict:
    """Layout counts per split and family, conductor-count range and the prior version of a shard dir."""
    from collections import Counter

    c, ns = Counter(), set()
    for d in read_shards(shard_dir):
        c.update(zip(d["split"].tolist(), d["family"].tolist()))
        ns.add(d["n"])
    s = read_schema(shard_dir)
    return {
        "layouts": sum(c.values()),
        "by_split": {sp: sum(v for (a, _), v in c.items() if a == sp) for sp in sorted({a for a, _ in c})},
        "by_family": {f"{a}/{b}": v for (a, b), v in sorted(c.items())},
        "n_range": [min(ns), max(ns)],
        "prior_version": s.get("physics", {}).get("prior_version"),
    }


def write_shards(rows: list[dict], out_dir, schema: dict, extra: dict | None = None) -> Path:
    """Write per-layout rows (keys of ARRAYS minus pairs, plus ``names``) as a shard dir; refuses duplicate keys."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    keys = [r["key"] for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"duplicate layout keys: {dup[:5]}")
    by_n: dict[int, list[dict]] = {}
    for r in rows:
        by_n.setdefault(len(r["sig"]), []).append(r)
    for n, rs in sorted(by_n.items()):
        z = {
            "key": np.array([r["key"] for r in rs]),
            "split": np.array([r["split"] for r in rs]),
            "family": np.array([r["family"] for r in rs]),
            "X": np.stack([r["X"] for r in rs]).astype(np.float32),
            "E": np.stack([r["E"] for r in rs]).astype(np.float32),
            "G": np.stack([r["G"] for r in rs]).astype(np.float32),
            "C": np.stack([r["C"] for r in rs]).astype(np.float32),
            "sig": np.stack([r["sig"] for r in rs]).astype(bool),
            "pri": np.stack([r["pri"] for r in rs]).astype(np.float32),
            "pairs": pairs_of(n),
        }
        np.savez_compressed(out / f"n{n:03d}.npz", **z)
    names = {r["key"]: list(r["names"]) for r in rows if r.get("names") is not None}
    if names:
        (out / "names.json").write_text(json.dumps(names), encoding="utf-8")
    s = dict(schema)
    s["shard_sha256"] = {p.name: sha256_file(p) for p in sorted(out.glob("n*.npz"))}
    if extra:
        s.update(extra)
    (out / "schema.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return out


def subset_shards(src, out_dir, train_keys=None, val_keys=None, keep=None) -> Path:
    """Copy of a shard dir restricted to ``train_keys`` (split set to train) and ``val_keys`` (split set to val), or to rows for
    which ``keep(shard_dict) -> bool mask`` is True (splits unchanged)."""
    src, out = Path(src), Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    tr, va = set(map(str, train_keys or ())), set(map(str, val_keys or ()))
    if tr & va:
        raise ValueError(f"keys both train and val: {sorted(tr & va)[:5]}")
    found = set()
    for d in read_shards(src):
        k = d["key"].astype(str)
        if keep is not None:
            m = np.asarray(keep(d), bool)
            split = d["split"]
        else:
            t, v = np.isin(k, list(tr)), np.isin(k, list(va))
            m = t | v
            split = np.where(t, "train", "val")
        found.update(k[m].tolist())
        if m.any():
            z = {a: (d[a] if a == "pairs" else d[a][m]) for a in d if a not in ("n", "file")}
            if keep is None:
                z["split"] = split[m]
            np.savez_compressed(out / d["file"], **z)
    if keep is None and found != tr | va:
        raise KeyError(f"keys not found in {src}: {sorted((tr | va) - found)[:5]}")
    s = read_schema(src)
    s["shard_sha256"] = {p.name: sha256_file(p) for p in sorted(out.glob("n*.npz"))}
    s["subset_of"] = str(src)
    (out / "schema.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    names = read_names(src)
    if names:
        (out / "names.json").write_text(json.dumps({k: v for k, v in names.items() if k in found}), encoding="utf-8")
    return out


def merge_shards(
    base_dir,
    user_dir,
    out_dir,
    user_split: str | None = None,
    all_split: str | None = None,
    allow_prior_mismatch: bool = False,
) -> Path:
    """Base shards + user shards -> one shard dir for collective retraining. Feature schemas and the prior version must agree
    (a user prior computed under another stack is refused); duplicate keys are refused. Base arrays the user rows lack (e.g.
    QCap's ``origin`` / ``group``) are filled with ``user`` / the row's family. ``user_split`` overrides the user rows'
    split, ``all_split`` every row's (``"train"`` for collective training)."""
    bs, us = read_schema(base_dir), read_schema(user_dir)
    for k in ("nodekeys", "edgekeys", "globkeys"):
        if list(bs[k]) != list(us[k]):
            raise ValueError(f"feature schema {k} differs between {base_dir} and {user_dir}")
    bp, up = bs.get("physics", {}), us.get("physics", {})
    if float(bp.get("alpha", np.nan)) != float(up.get("alpha", np.nan)):
        raise ValueError(f"prior alpha differs: base {bp.get('alpha')}, user {up.get('alpha')}")
    if bp.get("prior_version") != up.get("prior_version") and not allow_prior_mismatch:
        raise ValueError(
            f"prior version differs: base {bp.get('prior_version')!r}, user {up.get('prior_version')!r}. Build the user "
            "shards under the base's prior stack (or pass allow_prior_mismatch=True knowingly)"
        )
    base = {d["n"]: d for d in read_shards(base_dir)}
    user = {d["n"]: d for d in read_shards(user_dir)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    seen: set = set()
    for n in sorted(set(base) | set(user)):
        b, u = base.get(n), user.get(n)
        if u is not None:
            u = dict(u)
            if user_split is not None:
                u["split"] = np.full(len(u["key"]), user_split)
        if b is not None and u is not None:
            for k in set(b) - set(u) - {"n", "file"}:
                if b[k].dtype.kind != "U":
                    raise ValueError(f"base array {k!r} has no user counterpart and is not a string array")
                u[k] = u["family"].copy() if k == "group" else np.full(len(u["key"]), "user")
            extra = set(u) - set(b) - {"n", "file"}
            if extra:
                raise ValueError(f"user arrays {sorted(extra)} are not in the base shards")
            z = {k: (b[k] if k == "pairs" else np.concatenate([b[k], u[k]])) for k in b if k not in ("n", "file")}
        else:
            z = {k: v for k, v in (b or u).items() if k not in ("n", "file")}
        if all_split is not None:
            z["split"] = np.full(len(z["key"]), all_split)
        for k in z["key"].astype(str):
            if k in seen:
                raise ValueError(f"duplicate layout key {k!r} (base and user shards overlap)")
            seen.add(k)
        np.savez_compressed(out / f"n{n:03d}.npz", **z)
    s = dict(bs)
    s["shard_sha256"] = {p.name: sha256_file(p) for p in sorted(out.glob("n*.npz"))}
    s["merged"] = {
        "base": str(base_dir),
        "user": str(user_dir),
        "base_schema_sha256": sha256_file(Path(base_dir) / "schema.json"),
        "user_schema_sha256": sha256_file(Path(user_dir) / "schema.json"),
        "user_split": user_split,
        "all_split": all_split,
    }
    (out / "schema.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    names = {**read_names(base_dir), **read_names(user_dir)}
    if names:
        (out / "names.json").write_text(json.dumps(names), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------------------------- labels
def read_q3d_csv(path) -> tuple[list[str], np.ndarray, dict]:
    """Q3D ``cap_matrix.csv`` (one row per adaptive pass, columns ``C(a,b) [unit]``, also the ``C(a,b)*1e15 [pF]`` re-export)
    -> (net names, Maxwell matrix in fF of the last complete pass, info). Names: G first, then the file's order of first
    appearance."""
    rows = [r for r in csv.reader(open(path, encoding="utf-8")) if r]
    hdr, data = rows[0], rows[1:]
    cols, order = {}, []
    for j, h in enumerate(hdr):
        m = re.match(r"\s*C\(([^,]+),([^)]+)\)(\*1e15)?\s*\[(\w+)\]", h)
        if not m:
            continue
        a, b = m.group(1).strip(), m.group(2).strip()
        if m.group(4) not in TO_FF:
            raise ValueError(f"{path}: unknown unit {m.group(4)!r} in column {h!r}")
        cols[(a, b)] = (j, TO_FF[m.group(4)] * (1e-15 if m.group(3) else 1.0))
        for x in (a, b):
            if x not in order:
                order.append(x)
    if not cols:
        raise ValueError(f"{path}: no C(a,b) columns")
    names = ([GROUND] if GROUND in order else []) + [x for x in order if x != GROUND]
    n = len(names)
    missing = [(a, b) for a in names for b in names if (a, b) not in cols and (b, a) not in cols]
    if missing:
        raise ValueError(f"{path}: matrix entries missing from the export: {missing[:5]}")
    passes = []
    for r in data:
        C = np.full((n, n), np.nan)
        for ia, a in enumerate(names):
            for ib, b in enumerate(names):
                j, s = cols.get((a, b)) or cols[(b, a)]
                try:
                    C[ia, ib] = float(r[j]) * s
                except (ValueError, IndexError):
                    pass
        passes.append(C)
    complete = [k for k, C in enumerate(passes) if np.isfinite(C).all()]
    if not complete:
        raise LabelQCError(f"{path}: no complete adaptive pass")
    last = complete[-1]
    info = {"passes": len(passes), "label_pass_index": last, "complete_passes": len(complete)}
    if len(complete) >= 2:
        a, b = passes[complete[-2]], passes[last]
        big = np.abs(b) >= 0.1
        info["last_pass_change_max_pct"] = (
            float(np.max(np.abs(b - a)[big] / np.abs(b)[big]) * 100) if big.any() else 0.0
        )
    return names, passes[last], info


def label_qc(C, names, sym_rtol: float | None = None, sign_atol: float = 1e-6, closure_rtol: float = 1e-4) -> dict:
    """Label QC of a Maxwell matrix (fF, names G first). Returns the checks; raises LabelQCError on any failure."""
    from .qc import SYM_RTOL

    sym_rtol = SYM_RTOL if sym_rtol is None else sym_rtol
    C = np.asarray(C, float)
    n = len(names)
    fail = []
    if C.shape != (n, n):
        raise LabelQCError(f"matrix shape {C.shape} does not match {n} net names")
    if len(set(names)) != n:
        raise LabelQCError(f"duplicate net names {names}")
    if names[0] != GROUND:
        raise LabelQCError(f"ground net {GROUND!r} must be first (names {names})")
    if not np.isfinite(C).all():
        raise LabelQCError("non-finite entries in the label matrix")
    scale = float(np.abs(C).max())
    off = ~np.eye(n, dtype=bool)
    sym = float(np.abs(C - C.T).max() / max(scale, 1e-30))
    chk = {"symmetry_rel": sym}
    if sym > sym_rtol:
        fail.append(f"asymmetric (max |C - C^T| / max |C| = {sym:.3g} > {sym_rtol:g})")
    C = (C + C.T) / 2
    if not np.all(np.diag(C) > 0):
        fail.append("non-positive diagonal entries")
    chk["offdiag_max_fF"] = float(C[off].max()) if n > 1 else None
    if n > 1 and C[off].max() > sign_atol:
        fail.append(f"positive off-diagonal entry ({C[off].max():.3g} fF): not Maxwell signs")
    sig = np.array([x != GROUND for x in names])
    rows = C.sum(1)
    chk["closure_min_rel"] = float(np.min(rows[sig] / np.diag(C)[sig])) if sig.any() else None
    if sig.any() and np.min(rows[sig] / np.diag(C)[sig]) < -closure_rtol:
        fail.append(f"closure: a signal row sums below zero (min row sum / self = {chk['closure_min_rel']:.3g})")
    eig = float(np.linalg.eigvalsh(C[np.ix_(sig, sig)]).min()) if sig.any() else None
    chk["signal_block_min_eig_fF"] = eig
    if eig is not None and eig <= 0:
        fail.append(f"signal block not positive definite (min eigenvalue {eig:.3g} fF)")
    if fail:
        raise LabelQCError("; ".join(fail))
    chk["passed"] = True
    return chk


# ---------------------------------------------------------------------------------------------------------------- designs
@dataclass
class Design:
    """One layout and its label.

    gds: GDS path. labels: a ``cap_matrix.csv`` path (last complete pass), or a Maxwell matrix in fF with ``label_names``
    (any net order; ground ``G``), or None (inference-only rows are not allowed in training shards).
    names: how the layout's conductors are named, matched to the label names: ``"auto"`` (port-stub layouts, i.e. a (2, 0)
    marker: N / S fixed by the stubs and the pads named by the sibling design JSON's Q3D polygon map, the angle-slot or the
    file-order convention, whichever matches the label (``resolve_dataset_names``); else C1..Cn in file order),
    ``"dataset"`` (angle slots P<k>), ``"dataset_file_order"`` (N, S, P1, ... in file order), ``"design_json"``,
    ``"file_order"`` (C1..Cn), or an explicit list of signal names in file order.
    """

    gds: str
    labels: object = None
    label_names: list | None = None
    key: str | None = None
    family: str = "user"
    split: str = "train"
    names: object = "auto"
    ground_layers: tuple = ((1, 0),)
    conductor_layers: tuple = ((1, 10),)
    meta: dict = field(default_factory=dict)

    def resolved_key(self) -> str:
        if self.key:
            return str(self.key)
        p = Path(self.gds)
        return f"{p.parent.name}/{p.stem}"


def as_design(x) -> Design:
    if isinstance(x, Design):
        return x
    if isinstance(x, dict):
        return Design(**x)
    raise TypeError(f"cannot read a design from {type(x).__name__}: pass a Design or a dict of its fields")


def _has_marker(path, layer=(2, 0)) -> bool:
    import gdstk

    lib = gdstk.read_gds(str(path))
    cell = lib.top_level()[0]
    return any((p.layer, p.datatype) == layer for p in cell.polygons)


def design_json_names(gds) -> list[str] | None:
    """Signal net names in GDS file order from the design's sibling ``<stem>.json`` (the Q3D polygon map
    ``ansys_gds.q3d_signal.mapping_polygons`` + ``nets``), or None when there is no such map."""
    p = Path(gds).with_suffix(".json")
    if not p.is_file():
        return None
    try:
        q = json.loads(p.read_text(encoding="utf-8"))["ansys_gds"]["q3d_signal"]
        a2n = {v["assignment"]: k for k, v in q["nets"].items()}
        order = sorted(q["mapping_polygons"], key=lambda k: int(k.rsplit("_", 1)[-1]))
        names = [a2n[q["mapping_polygons"][k]] for k in order]
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    return [x for x in names if x != GROUND]


def resolve_dataset_names(signals, markers, label_names, json_names=None) -> tuple[list[str], str]:
    """Signal names of a port-stub layout (dataset convention) that match the label nets, and which naming gave them.

    N and S are fixed geometrically (N = body at the north stub (2, 0), S = body at the south stub (3, 0) or the origin).
    Candidates, each kept only if its names are unique, equal the label's signal nets and put N and S on those bodies:
    ``design_json`` (the Q3D polygon map of the sibling design JSON: exp13-15, quantum-metal exports), ``dataset`` (pads
    P<k> by angle slot: the multi-pad star, exp12) and ``dataset_file_order`` (N, S, P1, P2, ... in file order). The design
    JSON wins when valid; otherwise one valid candidate (or two that agree) is used, and anything else is refused."""
    from ._core._tlsq.uge.scene import tlsq_namer

    want = sorted(x for x in label_names if x != GROUND)
    geo = tlsq_namer(signals, markers)
    n_i, s_i = geo.index("N"), geo.index("S")
    cands = {}
    if json_names is not None:
        cands["design_json"] = list(json_names)
    cands["dataset"] = geo
    cands["dataset_file_order"] = ["N", "S"] + [f"P{k}" for k in range(1, len(signals) - 1)]

    def ok(c):
        return (
            len(c) == len(signals)
            and sorted(c) == want
            and len(set(c)) == len(c)
            and "N" in c
            and "S" in c
            and c.index("N") == n_i
            and c.index("S") == s_i
        )

    valid = {k: c for k, c in cands.items() if ok(c)}
    if "design_json" in valid:
        return valid["design_json"], "design_json"
    if not valid:
        raise LabelQCError(
            f"no naming of the layout bodies matches the label nets {want} ({cands}); pass names= (a list in file order); "
            "nets are never merged"
        )
    if len(valid) == 2 and valid["dataset"] != valid["dataset_file_order"]:
        raise LabelQCError(
            f"ambiguous naming: {valid} both match the label nets {want}; pass names='dataset', 'dataset_file_order' "
            "or a list in file order"
        )
    k = "dataset" if "dataset" in valid else "dataset_file_order"
    return valid[k], k


def scene_of(d: Design, label_names=None):
    """(Scene, naming used) for a design: the training geometry (no crop), ground = net 0. ``label_names`` lets
    ``names="auto"`` pick the dataset naming that matches the label (see ``resolve_dataset_names``)."""
    from ._core._tlsq.uge.scene import scene_from_gds, tlsq_namer

    mode = d.names
    if mode == "auto":
        mode = "dataset" if _has_marker(d.gds) else "file_order"
        if mode == "dataset" and label_names is not None:
            used = {}
            jn = design_json_names(d.gds)

            def auto_namer(S, m):
                names, used["k"] = resolve_dataset_names(S, m, label_names, jn)
                return names

            sc = scene_from_gds(d.gds, namer=auto_namer)
            return sc, f"auto:{used['k']}"
    if mode == "dataset":
        return scene_from_gds(d.gds, namer=tlsq_namer), "dataset"
    if mode in ("dataset_file_order", "design_json"):
        if mode == "design_json":
            sig = design_json_names(d.gds)
            if sig is None:
                raise LabelQCError(f"{d.gds}: no design JSON with a Q3D polygon map next to the GDS")
        else:
            sig = None

        def ordered(S, m):
            ns = sig if sig is not None else ["N", "S"] + [f"P{k}" for k in range(1, len(S) - 1)]
            if len(S) != len(ns):
                raise LabelQCError(f"{d.gds}: {len(S)} signal polygons but {len(ns)} names ({mode})")
            return list(ns)

        return scene_from_gds(d.gds, namer=ordered), mode
    if isinstance(mode, (list, tuple)):
        sig = [str(x) for x in mode if x != GROUND]

        def namer(S, m):
            if len(S) != len(sig):
                raise LabelQCError(f"{d.gds}: {len(S)} signal polygons but {len(sig)} names given")
            return sig

        return scene_from_gds(d.gds, namer=namer), "explicit"
    if mode != "file_order":
        raise ValueError(
            f"names={mode!r}: use 'auto', 'dataset', 'dataset_file_order', 'design_json', 'file_order' or a list of names"
        )
    from ._core import api as A
    from ._runtime import prepare_layout

    lay = prepare_layout(d.gds, ground_layers=d.ground_layers, conductor_layers=d.conductor_layers)
    return A.scene_from_prepared(lay), "file_order"


def labels_of(d: Design) -> tuple[list[str], np.ndarray, dict]:
    if d.labels is None:
        raise LabelQCError(f"{d.resolved_key()}: no labels (training rows need a Maxwell matrix)")
    if isinstance(d.labels, (str, os.PathLike)):
        names, C, info = read_q3d_csv(d.labels)
        return names, C, {"source": str(d.labels), **info}
    if d.label_names is None:
        raise LabelQCError(f"{d.resolved_key()}: a label matrix needs label_names")
    return [str(x) for x in d.label_names], np.asarray(d.labels, float), {"source": "matrix"}


def _stack_registry():
    from .models import registry_view

    return registry_view({}, ".", "squadds-training-data")


def embed_design(d: Design, threads: int = 2, prior: str = "optimized", schema: dict | None = None) -> dict:
    """Shard row of one design: embedding (exact, no far-sample decimation), prior solves at 1.5 and 3 panels per gap under
    the corrected stack, and the QC-checked label reordered to the scene's net order (G first)."""
    from . import _runtime as RT
    from ._core import api as A
    from ._core import paper as PA
    from ._core._tlsq.uge import blocks as B
    from .predictor import quiet

    if prior != "optimized":
        raise NotImplementedError(
            f"prior={prior!r}: training shards carry the optimized_physics_prior solves (1.5, 3); other priors are TODO"
        )
    schema = schema or base_schema()
    key = d.resolved_key()
    lnames, Cl, linfo = labels_of(d)
    if GROUND not in lnames:
        raise LabelQCError(f"{key}: the label has no ground net {GROUND!r} (names {lnames})")
    lnames_g = [GROUND] + [x for x in lnames if x != GROUND]
    perm = [lnames.index(x) for x in lnames_g]
    try:
        qc = label_qc(Cl[np.ix_(perm, perm)], lnames_g)
    except LabelQCError as e:
        raise LabelQCError(f"{key}: label QC failed: {e}") from e
    t0 = time.perf_counter()
    sc, naming = scene_of(d, lnames)
    names = list(sc.names()) if callable(getattr(sc, "names", None)) else [c.name for c in sc.conductors]
    if sorted(names) != sorted(lnames):
        raise LabelQCError(
            f"{key}: layout nets {names} do not match the label nets {lnames} (naming {naming!r}); nets are never merged"
        )
    old = dict(B.FAR_DECIMATION)
    B.FAR_DECIMATION.update(k=0, d_far=100.0)
    try:
        from ._core._tlsq.uge.graph import scene_graph

        row = PA.lt.add_tokens(sc, scene_graph(sc))
    finally:
        B.FAR_DECIMATION.clear()
        B.FAR_DECIMATION.update(old)
    t1 = time.perf_counter()
    if list(row["names"]) != names:
        raise RuntimeError(f"{key}: embedding names {row['names']} differ from the scene {names}")
    with RT.LOCK, quiet():
        threads = RT.configure(threads)
        PA.apply_stack(_stack_registry(), {"stack": "corrected"}, threads)
        P, tl = PA.solve_levels(sc, list(PRIOR_LEVELS), threads)
        stack = {"name": "corrected", "patches": list(A._STATE.get("patches", [])), "mode": A._STATE.get("mode")}
    t2 = time.perf_counter()
    n = len(names)
    pr = pairs_of(n)
    nk, ek, gk = schema["nodekeys"], schema["edgekeys"], schema["globkeys"]
    assert nk[-1] == "missing_log_min_halfwidth"
    wi = nk.index("log_min_halfwidth")
    X = np.array([[row["nodes"][a][k] for k in nk[:-1]] for a in range(n)], np.float32)
    X = np.concatenate([X, (~np.isfinite(X[:, wi])).astype(np.float32)[:, None]], -1)
    E = np.array([[row["edges"][(int(i), int(j))][k] for k in ek] for i, j in pr], np.float32).reshape(len(pr), len(ek))
    G = np.array([row["global"][k] for k in gk], np.float32)
    pri = np.stack([np.asarray(P[p], float)[pr[:, 0], pr[:, 1]] for p in PRIOR_LEVELS], -1).astype(np.float32)
    if not (np.isfinite(E).all() and np.isfinite(G).all() and np.isfinite(np.delete(X, wi, axis=-1)).all()):
        raise ValueError(f"{key}: non-finite embedding features")
    if not (np.isfinite(pri).all() and (pri > 0).all()):
        raise ValueError(f"{key}: prior mutuals not finite and positive")
    idx = [lnames.index(x) for x in names]
    C = np.asarray(Cl, float)[np.ix_(idx, idx)]
    C = (C + C.T) / 2
    return {
        "key": key,
        "split": d.split,
        "family": d.family,
        "names": names,
        "X": X,
        "E": E,
        "G": G,
        "C": C.astype(np.float32),
        "sig": np.array([x != GROUND for x in names]),
        "pri": pri,
        "info": {
            "gds": str(d.gds),
            "naming": naming,
            "label": linfo,
            "qc": qc,
            "stack": stack,
            "timing_s": {"embedding": t1 - t0, "prior": t2 - t1, "prior_levels": tl},
            **({"meta": d.meta} if d.meta else {}),
        },
    }


def _embed_worker(args):
    d, threads, prior, on_error = args
    try:
        return embed_design(d, threads=threads, prior=prior), None
    except (LabelQCError, ValueError) as e:
        if on_error == "raise":
            raise
        return None, {"key": d.resolved_key(), "gds": str(d.gds), "reason": str(e)}


def build_shards(
    designs, out_dir, prior: str = "optimized", threads: int = 2, workers: int = 1, on_error: str = "raise"
) -> dict:
    """User designs -> a P2d shard dir (``n0NN.npz`` + ``schema.json`` + ``names.json`` + ``build.json``).

    designs: Design objects or dicts of their fields. Labels are QC-checked and matched to the layout nets by name; a design
    that fails is refused (``on_error="raise"``, default) or skipped and listed in ``build.json`` (``"skip"``). Priors are
    computed on this machine (macOS is not bit-identical to the released x86 priors: ~2.4e-3 relative at most); ``threads``
    per design, ``workers`` designs at a time (spawned processes)."""
    from . import _runtime as RT

    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")
    ds = [as_design(x) for x in designs]
    if not ds:
        raise ValueError("no designs")
    schema = base_schema()
    t = time.perf_counter()
    jobs = [(d, threads, prior, on_error) for d in ds]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with ProcessPoolExecutor(min(workers, len(ds)), mp_context=get_context("spawn")) as ex:
            res = list(ex.map(_embed_worker, jobs))
    else:
        res = [_embed_worker(j) for j in jobs]
    rows = [r for r, e in res if r is not None]
    refused = [e for r, e in res if e is not None]
    if not rows:
        raise LabelQCError(f"every design was refused: {refused[:3]}")
    from squadds import __version__

    build = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": RT.machine(),
        "platform": platform.platform(),
        "threads": threads,
        "workers": workers,
        "package": f"squadds {__version__}",
        "code_sha256": sha256_file(__file__),
        "layouts": len(rows),
        "refused": refused,
        "seconds": time.perf_counter() - t,
        "note": "priors computed on this machine; released shards use HPC x86 priors",
    }
    out = write_shards(rows, out_dir, schema, extra={"build": {k: v for k, v in build.items() if k != "refused"}})
    build["designs"] = {r["key"]: r["info"] for r in rows}
    (out / "build.json").write_text(json.dumps(build, indent=1, default=str), encoding="utf-8")
    return {"out_dir": str(out), "layouts": len(rows), "refused": refused, "seconds": build["seconds"]}


# ---------------------------------------------------------------------------------------------------------------- dataset
def shards_from_model_input(model_input, labels, out_dir, splits=("train", "val")) -> Path:
    """HOOK: the dataset's ``model_input`` table (``squadds.datasets.planar_capacitance_schema``, one row per layout: flat
    X / E / G / pri / sig with ``net_order``) and its ``labels`` table (long format, ``a_idx <= b_idx``) -> a shard dir.
    Both arguments: pandas DataFrames or parquet paths. Test rows are never read (``splits`` may not contain ``test``)."""
    import pandas as pd

    if "test" in splits:
        raise PermissionError("test labels are not training data")
    mi = pd.read_parquet(model_input) if isinstance(model_input, (str, os.PathLike)) else model_input
    lab = pd.read_parquet(labels) if isinstance(labels, (str, os.PathLike)) else labels
    mi = mi[mi["split"].isin(splits)]
    schema = base_schema()
    versions = set(mi["prior_version"].unique())
    if len(versions) > 1:
        raise ValueError(f"model_input mixes prior versions {sorted(versions)}")
    nx, ne, ng = len(schema["nodekeys"]), len(schema["edgekeys"]), len(schema["globkeys"])
    lab = lab[lab["design_id"].isin(set(mi["design_id"]))]
    groups = {k: g for k, g in lab.groupby("design_id")}
    rows = []
    for r in mi.itertuples(index=False):
        n = int(r.n_conductors)
        p = n * (n - 1) // 2
        g = groups.get(r.design_id)
        if g is None:
            raise KeyError(f"{r.design_id}: no labels")
        C = np.full((n, n), np.nan)
        C[g["a_idx"].to_numpy(), g["b_idx"].to_numpy()] = g["C_fF"].to_numpy()
        C[g["b_idx"].to_numpy(), g["a_idx"].to_numpy()] = g["C_fF"].to_numpy()
        names = list(r.net_order)
        label_qc(C, names)
        rows.append(
            dict(
                key=r.design_id,
                split=r.split,
                family=r.model_family,
                names=names,
                X=np.asarray(r.X, np.float32).reshape(n, nx),
                E=np.asarray(r.E, np.float32).reshape(p, ne),
                G=np.asarray(r.G, np.float32).reshape(ng),
                C=C,
                sig=np.asarray(r.sig, bool),
                pri=np.asarray(r.pri, np.float32).reshape(p, 2),
            )
        )
    v = next(iter(versions), None)
    if v is not None and not str(v).startswith(schema["physics"]["prior_version"]):
        schema["physics"]["prior_version"] = str(v)
    return write_shards(rows, out_dir, schema, extra={"source": "dataset model_input table"})
