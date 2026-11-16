"""Paths, readers and writers shared by the SQuADDS_PlanarCapacitance builders.

Readers are adapted from tl-squadds `scripts/hfds_build.py` (the v1 builder); tl-squadds itself is only read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SQ = HERE.parents[2]
sys.path.insert(0, str(SQ))
from squadds.datasets import planar_capacitance_schema as SCH  # noqa: E402

R = Path(os.environ.get("TLSQ_ROOT", "/Users/shanto/LFL/tl_squadds"))
DL = R / "data_local"
ART = Path(os.environ.get("SQML_ARTIFACTS", "/Users/shanto/LFL/squadds_ml_artifacts"))
OUT = ART / "dataset" / "SQuADDS_PlanarCapacitance"
HELD = ART / "dataset" / "SQuADDS_PlanarCapacitance-heldout"
WORK = ART / "dataset" / "_work"
HPC_RAW = ART / "hpc_raw" / "q3d_campaign"
V1 = DL / "release_core"
C_HPC = "/project2/shaas_31/shanto/tlsq_store/results/q3d_campaign"
P_HPC = "/project2/shaas_31/shanto/palace_pipeline/designs/q3d_campaign"
DSV = SCH.DATASET_VERSION
PROV = dict(source="reference", licence_id=SCH.LICENSE, dataset_version_added=DSV)
# author 2026-10-29: the exp6-8 in-family test labels are already public in SQuADDS_DB, so they ship in the public repo
PUBLIC_TEST_COLLECTIONS = ("exp6", "exp7", "exp8")
PUBLIC_TEST_REASON = "labels public: already public in SQuADDS/SQuADDS_DB"
TRAIN_STACK = dict(substrate_material="cryo_silicon", eps_r=11.45, substrate_um=350.0, tan_delta=1e-7)
# Hugging Face rejects a folder with more than 10,000 files: these collections keep their GDS in sub-folders of 5,000 by id
BUCKETED_GDS, GDS_BUCKET, HF_MAX_FILES_PER_DIR = ("exp8", "exp9"), 5000, 10000


def gds_relpath(design_id: str) -> str:
    """Repo path of a design's GDS: ``gds/<collection>/<name>.gds``; exp8/exp9 ``gds/<collection>/<id // 5000>/<name>.gds``."""
    coll, name = design_id.removeprefix("ref/").split("/", 1)
    if coll in BUCKETED_GDS:
        return f"gds/{coll}/{int(name.rsplit('_', 1)[1]) // GDS_BUCKET}/{name}.gds"
    return f"gds/{coll}/{name}.gds"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(HERE.glob("*.py")) + [SQ / "squadds/datasets/planar_capacitance_schema.py"]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def jmin(o) -> str:
    return json.dumps(o, separators=(",", ":"), default=str)


def gds_info(path: str) -> dict:
    """File sha256, canonical polygon hash (1 nm), header, counts, bbox, conductor count on the dataset layer map."""
    import gdstk

    lib = gdstk.read_gds(path)
    top = lib.top_level()
    s = lib.unit / 1e-9
    polys = [p for c in top for p in c.polygons]
    key = sorted((p.layer, p.datatype, np.round(p.points * s).astype(np.int64).tobytes()) for p in polys)
    h = hashlib.sha256()
    for la, dt, b in key:
        h.update(f"{la}/{dt}:".encode())
        h.update(b)
    layers = sorted({(p.layer, p.datatype) for p in polys})
    allpts = np.vstack([p.points for p in polys]) * lib.unit * 1e6 if polys else np.zeros((1, 2))
    lo, hi = allpts.min(0), allpts.max(0)
    fs = sha256_file(path)
    return dict(
        layout_id="sha256:" + fs,
        sha256=fs,
        geom_hash="sha256:" + h.hexdigest(),
        unit_m=lib.unit,
        precision_m=lib.precision,
        top_cell=",".join(c.name for c in top),
        layers=layers,
        n_polygons=len(polys),
        n_vertices=int(sum(len(p.points) for p in polys)),
        bbox_um=[float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        file_bytes=os.path.getsize(path),
        n_cond_layout=1 + sum(1 for p in polys if p.layer == 1 and p.datatype != 0),
    )


def layer_map(layers) -> str:
    ls = [list(x) for x in layers]
    if any(la[0] == 1 and la[1] not in (0, 10) for la in ls):
        return jmin({"ground": [[1, 0]], "conductor": [la for la in ls if la[0] == 1 and la[1] != 0]})
    return jmin({"ground": [[1, 0]], "conductor": [[1, 10]], "ports": [la for la in ls if la[0] != 1]})


def read_csv_passes(path):
    """Any Q3D cap_matrix CSV -> (passes, nets, C[n_pass, N, N] fF, freq_GHz); NaN pre-allocated rows skipped."""
    rows = [r for r in csv.reader(open(path)) if r]
    hdr = rows[0]
    good = [r for r in rows[1:] if all(x.strip() and math.isfinite(float(x)) for x in r[2:])]
    ent = []
    for j, h in enumerate(hdr[2:], start=2):
        a, b, x15, unit = re.match(r"C\((\w+),(\w+)\)(\*1e15)? \[(\w+)\]", h).groups()
        ent.append((a, b, j, {"pF": 1e3, "fF": 1.0}[unit] * (1e-15 if x15 else 1.0)))
    nets = list(dict.fromkeys(a for a, *_ in ent))
    ix = {n: i for i, n in enumerate(nets)}
    C = np.full((len(good), len(nets), len(nets)), np.nan)
    for a, b, j, sc in ent:
        C[:, ix[a], ix[b]] = [float(r[j]) * sc for r in good]
    freq = float(good[-1][0]) if good else float("nan")
    return np.array([int(float(r[1])) for r in good]), nets, C, freq


def read_conv(path) -> dict:
    if not path or not os.path.exists(path):
        return {}
    t = open(path).read()

    def g(k, f=float):
        m = re.search(rf"{k}\s*:\s*(\S+)", t)
        try:
            return f(m.group(1)) if m else None
        except ValueError:
            return None

    hist = []
    for m in re.finditer(r"^\s*(\d+)\|\s*(\d+)\|\s*([^|]+)\|", t, re.M):
        d = m.group(3).strip()
        hist.append(
            {"pass": int(m.group(1)), "n_triangles": int(m.group(2)), "delta_pct": None if d == "N/A" else float(d)}
        )
    out = {
        "pass_history": hist,
        "passes_completed": g("Completed", int),
        "delta_target_pct": g("Target"),
        "delta_pct_final": g("Current"),
    }
    if out["delta_target_pct"] is not None and out["delta_pct_final"] is not None:
        out["converged"] = out["delta_pct_final"] <= out["delta_target_pct"]
    return out


def qc_flags(C: np.ndarray, nets: list[str], Cprev: np.ndarray | None, sym_rel: float | None = None) -> dict:
    """Label QC of one final matrix (rules of tl-squadds flagship_v4/label_qc.py). C in `nets` order (any), fF.

    Matrices without self entries (lattice exports) skip closure and PD (recorded as None)."""
    n = len(nets)
    sig = np.array([v != "G" for v in nets])
    off = ~np.eye(n, dtype=bool)
    Cz = np.where(np.isfinite(C), C, 0.0)
    if sym_rel is None:
        sym_rel = float(np.max(np.abs(Cz - Cz.T)) / max(1e-300, np.max(np.abs(np.diag(Cz)))))
    Cs = 0.5 * (Cz + Cz.T)
    has_diag = bool(np.isfinite(np.diag(C)).all())
    q = dict(
        qc_sym_rel=sym_rel,
        qc_symmetric=bool(sym_rel < 1e-5),
        qc_signs=bool(np.nanmax(np.where(off, C, -np.inf)) <= 1e-6 and (not has_diag or (np.diag(C) > 0).all())),
    )
    if has_diag and sig.any():
        rs = Cs.sum(1)[sig]
        q["qc_closure_max_rel"] = float(np.max(np.abs(rs) / np.diag(Cs)[sig]))
        q["qc_closure"] = bool(np.min(rs) >= -1e-6)
        q["qc_min_eig_signal_fF"] = float(np.linalg.eigvalsh(Cs[np.ix_(sig, sig)]).min())
        q["qc_pd"] = bool(q["qc_min_eig_signal_fF"] > 0)
    else:
        q.update(qc_closure_max_rel=None, qc_closure=None, qc_min_eig_signal_fF=None, qc_pd=None)
    chg = None
    if Cprev is not None:
        P = 0.5 * (np.where(np.isfinite(Cprev), Cprev, 0.0) + np.where(np.isfinite(Cprev), Cprev, 0.0).T)
        m = off & (Cs != 0) & np.isfinite(C)
        chg = float(np.max(np.abs(P[m] / Cs[m] - 1)) * 100) if m.any() else None
    q["qc_last_pass_change_max_pct"] = chg
    return q


def qc_convergence(sim: dict) -> str:
    st = sim.get("status")
    if st == "incomplete":
        return "truncated"
    d, t = _finite(sim.get("delta_pct_final")), _finite(sim.get("delta_target_pct"))
    if d is not None and t is not None:
        return "converged" if d <= t else "not_converged"
    return "unrecorded"


def _finite(v):
    """A float, or None for a missing or NaN record (exp6-12 exports carry no Delta % record)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def entry_kind(a, b):
    if a == b:
        return "self_ground" if a == "G" else "self_signal"
    pad, gnd = a.startswith("P") or b.startswith("P"), "G" in (a, b)
    return ("pad_ground" if gnd else "pad_other") if pad else ("signal_ground" if gnd else "signal_signal")


def write_parquet(table, path: Path, entry: bool, meta: dict | None = None):
    """zstd 3, dictionary strings, byte-stream-split floats, delta ints (PLAN_DATASET_HF.md 1.4)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    dict_cols = [f.name for f in table.schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    enc = {}
    for f in table.schema:
        if f.name in dict_cols:
            continue
        if pa.types.is_floating(f.type):
            enc[f.name] = "BYTE_STREAM_SPLIT"
        elif pa.types.is_integer(f.type):
            enc[f.name] = "DELTA_BINARY_PACKED"
    md = {
        b"squadds.schema_version": SCH.SCHEMA_VERSION.encode(),
        b"squadds.dataset_version": DSV.encode(),
        b"squadds.build_code_sha256": code_sha().encode(),
    }
    for k, v in (meta or {}).items():
        md[k.encode()] = str(v).encode()
    table = table.replace_schema_metadata(md)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=3,
        use_dictionary=dict_cols,
        column_encoding=enc,
        row_group_size=65536 if entry else 8192,
        write_statistics=True,
    )


def to_table(df, name: str):
    """pandas -> pyarrow with the declared schema of `name` (column order, types)."""
    import pyarrow as pa

    sch = SCH.arrow_schema(name)
    missing = [c for c in sch.names if c not in df.columns]
    assert not missing, (name, missing)
    return pa.Table.from_pandas(df[sch.names], schema=sch, preserve_index=False)


def key_to_design_id(key: str) -> str:
    """Shard / X1 key -> design_id."""
    if key.startswith("grid_qubit_"):
        return f"ref/grid/{key}"
    a, b = key.split("/", 1)
    coll = {"random": "random_v1", "wonky": "wonky_v1", "qm": "qm116", "p3rand": "p3rand36"}.get(a, a)
    name = b if b.startswith("q3d_cap_") else f"q3d_cap_{b}"
    return f"ref/{coll}/{name}"
