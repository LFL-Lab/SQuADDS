"""Checks on the written files: Table I counts, labels vs the tlsq reader, v1 subset, no public test labels, processed values
bit-equal to the shard sources, labels vs the shard / X1 matrices, sizes."""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import DL, HELD, OUT, PUBLIC_TEST_COLLECTIONS, V1, R, gds_relpath, key_to_design_id, sha256_file

PUB = OUT / "data"


def _read(pattern, columns=None, root=PUB, filters=None):
    fs = sorted(glob.glob(str(root / pattern)))
    return (
        pd.concat([pd.read_parquet(f, columns=columns, filters=filters) for f in fs], ignore_index=True)
        if fs
        else pd.DataFrame()
    )


def table_i():
    tex = (R / "paper_v4/latex_rev2_shanto/floats/tab_data.tex").read_text()
    macros = {}
    for f in glob.glob(str(R / "paper_v4/latex_rev2_shanto/**/*.tex"), recursive=True):
        for m in re.finditer(r"\\newcommand\{?\\(\w+)\}?\{([\d{},]+)\}", open(f).read()):
            macros[m.group(1)] = m.group(2)
    rows = {}
    for line in tex.splitlines():
        m = re.match(r"\s*([^&\\]+?)\s*&\s*([^&]*)&\s*([^&]*)&\s*([^&]*)&\s*([^\\]*)\\\\", line)
        if not m or m.group(1).startswith("design family"):
            continue

        def num(s):
            s = s.strip()
            for k, v in macros.items():
                s = s.replace(f"\\{k}{{}}", v).replace(f"\\{k}", v)
            s = s.replace("{,}", "").replace(",", "").replace("{", "").replace("}", "")
            return int(s) if s.isdigit() else None

        rows[m.group(1).replace("$N$", "N").strip()] = tuple(num(m.group(k)) for k in (3, 4, 5))
    return rows


def check_counts():
    D = _read("designs/designs-*.parquet", ["design_id", "collection", "paper_family", "split", "labelled"])
    t1 = table_i()
    name = {"interdigital (3 sweeps)": "interdigital", "large N": "large N"}
    out, ok = [], True
    for row, (tr, va, te) in t1.items():
        fam = name.get(row, row)
        if row.startswith("\\quad") or row == "total":
            continue
        g = D[D.paper_family == fam]
        got = tuple(int((g.split == s).sum()) for s in ("train", "val", "test"))
        exp = tuple(x or 0 for x in (tr, va, te))
        out.append(dict(table_i_row=row, table_i=exp, dataset=got, match=got == exp))
        ok &= got == exp
    tot = tuple(int((D.split == s).sum()) for s in ("train", "val", "test"))
    out.append(dict(table_i_row="total", table_i=t1.get("total"), dataset=tot, match=tot == t1.get("total")))
    lab = D[D.split.isin(["train", "val"])]
    return dict(
        rows=out,
        all_rows_match=bool(ok and tot == t1.get("total")),
        train_val_without_label=int((~lab.labelled).sum()),
        test_labelled=int(D[(D.split == "test")].labelled.sum()),
        test_designs=int((D.split == "test").sum()),
    )


def _tlsq(args):
    exp, name = args
    os.environ.setdefault("TLSQ_DATA", str(DL / "designs_ood/data"))
    sys.path.insert(0, str(R / "src"))
    from tlsq.designs import Design
    from tlsq.labels import read_cap_matrix

    d = Design(exp, name)
    if not (d.design_dir / ("cap_matrix_full.csv" if exp == "exp12" else "cap_matrix.csv")).exists():
        return exp, name, None, None
    _, nets, C = read_cap_matrix(d)
    return exp, name, nets, C[-1]


def check_exp612(workers):
    D = _read("designs/designs-*.parquet", ["design_id", "collection"])
    D = D[D.collection.str.match(r"exp([6-9]|1[0-2])$")]
    L = pd.concat(
        [
            _read("labels/labels-*.parquet", ["design_id", "net_a", "net_b", "C_fF"]),
            _read("labels/labels-test.parquet", ["design_id", "net_a", "net_b", "C_fF"], root=HELD / "data"),
        ]
    )
    L = L[L.design_id.isin(set(D.design_id))]
    lab = {(d, a, b): c for d, a, b, c in zip(L.design_id, L.net_a, L.net_b, L.C_fF)}
    jobs = [(d.split("/")[1], d.split("/")[2]) for d in D.design_id]
    mx, n, nlab, miss = 0.0, 0, 0, []
    with Pool(workers) as pool:
        for exp, name, nets, C in pool.imap_unordered(_tlsq, jobs, chunksize=128):
            did = f"ref/{exp}/{name}"
            if nets is None:
                continue
            n += 1
            for i, a in enumerate(nets):
                for j, b in enumerate(nets):
                    v = lab.get((did, a, b))
                    if v is None:
                        continue
                    nlab += 1
                    mx = max(mx, abs(v - C[i, j]))
            if not any((did, a, a) in lab for a in nets):
                miss.append(did)
    return dict(
        designs_read=n,
        entries_compared=nlab,
        max_abs_diff_fF=mx,
        labelled_by_tlsq_but_no_label=miss[:20],
        n_missing=len(miss),
    )


def check_superset():
    out = {}
    v1d = pd.concat(
        [
            pd.read_parquet(f, columns=["design_id", "layout_id"])
            for f in glob.glob(str(V1 / "data/v1/designs/*.parquet"))
        ]
    )
    D = _read("designs/designs-*.parquet", ["design_id", "layout_id"])
    out["designs_v1"], out["designs_missing"] = len(v1d), int((~v1d.design_id.isin(D.design_id)).sum())
    lid = dict(zip(D.design_id, D.layout_id))
    out["designs_layout_id_changed"] = int(sum(lid.get(d) != lo for d, lo in zip(v1d.design_id, v1d.layout_id)))
    bad = 0
    for p in (V1 / "gds").rglob("*.gds"):
        q = OUT / gds_relpath(p.relative_to(V1 / "gds").with_suffix("").as_posix())
        bad += (not q.exists()) or sha256_file(q) != sha256_file(p)
    out["gds_v1"], out["gds_missing_or_different"] = sum(1 for _ in (V1 / "gds").rglob("*.gds")), int(bad)
    v1l = pd.read_parquet(V1 / "data/v1/layouts/layouts.parquet", columns=["layout_id"])
    LY = _read("layouts/layouts.parquet", ["layout_id"])
    out["layouts_missing"] = int((~v1l.layout_id.isin(LY.layout_id)).sum())
    v1s = pd.concat(
        [pd.read_parquet(f, columns=["simulation_id"]) for f in glob.glob(str(V1 / "data/v1/simulations/*.parquet"))]
    )
    S = _read("simulations/simulations-*.parquet", ["simulation_id"])
    out["simulations_missing"] = int((~v1s.simulation_id.isin(S.simulation_id)).sum())
    for t, key in (
        ("labels", ["simulation_id", "net_a", "net_b"]),
        ("label_passes", ["simulation_id", "net_a", "net_b", "pass"]),
    ):
        a = pd.concat(
            [
                pd.read_parquet(f, columns=key + ["C_fF"])
                for f in glob.glob(str(V1 / f"data/v1/{t}/*.parquet"))
                + glob.glob(str(V1 / f"private_heldout/data/v1/{t}/*.parquet"))
            ]
        )
        b = pd.concat(
            [
                _read(f"{t}/{t}-*.parquet", key + ["C_fF"]),
                _read(f"{t}/{t}-*.parquet", key + ["C_fF"], root=HELD / "data"),
            ]
        )
        m = a.merge(b, on=key, how="left", suffixes=("_v1", "_new"))
        out[f"{t}_v1_rows"] = len(a)
        out[f"{t}_missing"] = int(m.C_fF_new.isna().sum())
        out[f"{t}_value_differs"] = int((m.C_fF_new.notna() & (m.C_fF_new != m.C_fF_v1)).sum())
    out["v1_subset_of_new"] = all(
        out[k] == 0 for k in out if k.endswith(("missing", "differs", "changed", "different", "differing"))
    )
    return out


def check_no_test_labels():
    D = _read("designs/designs-*.parquet", ["design_id", "split", "geom_hash", "collection"])
    pub = (D.split == "test") & D.collection.isin(PUBLIC_TEST_COLLECTIONS)
    public_test = set(D.loc[pub, "design_id"])
    test = set(D.loc[(D.split == "test") & ~pub, "design_id"])
    th = set(D.loc[(D.split == "test") & ~pub, "geom_hash"].dropna())
    same_geom = set(D.loc[D.geom_hash.isin(th), "design_id"])
    out = {}
    for t in ("labels", "label_passes"):
        x = _read(f"{t}/{t}-*.parquet", ["design_id", "split"])
        out[f"public_{t}_rows"] = len(x)
        out[f"public_{t}_exp6_8_infamily_test_rows"] = int(x.design_id.isin(public_test).sum())
        out[f"public_{t}_test_design_rows"] = int(x.design_id.isin(test).sum())
        out[f"public_{t}_test_geometry_rows"] = int(x.design_id.isin(same_geom).sum())
        out[f"public_{t}_split_test_rows"] = int(((x.split == "test") & ~x.design_id.isin(public_test)).sum())
    for f in glob.glob(str(PUB / "**/*.parquet"), recursive=True):
        cols = pq.read_schema(f).names
        if "model_input" in f or "geometric_embedding" in f or "prior" in f:
            assert not {"C", "C_fF_label", "target"} & set(cols), f
    H = _read("labels/labels-test.parquet", ["design_id"], root=HELD / "data")
    out["heldout_label_designs"] = int(H.design_id.nunique())
    out["test_designs"] = len(test) + len(public_test)
    out["n_public_test_designs_exp6_8"] = len(public_test)
    out["test_designs_with_heldout_label"] = int(len(test & set(H.design_id)))
    out["exp6_8_test_designs_with_heldout_label"] = int(len(public_test & set(H.design_id)))
    # rule (author 2026-10-29): no public test labels outside the exp6-8 in-family test (already public in SQuADDS_DB)
    out["no_public_test_labels_outside_exp6_8"] = all(
        v == 0
        for k, v in out.items()
        if k.startswith("public_")
        and not k.endswith("_rows")
        or k.endswith(("test_design_rows", "test_geometry_rows", "split_test_rows"))
    )
    return out


def check_processed():
    """Written embedding / prior / model_input values == the shard sources (bit for bit)."""
    from processed import SF, _x1_records

    names = {}
    pi = pickle.load(open(DL / "multifid/pool_index.pkl", "rb"))
    names.update({k: v["names"] for k, v in pi["src"].items()})
    nodes = _read("geometric_embedding/nodes-*.parquet")
    edges = _read("geometric_embedding/edges-*.parquet")
    glob_ = _read("geometric_embedding/globals-*.parquet")
    opt = _read(
        "optimized_physics_prior/optimized_physics_prior-*.parquet",
        ["design_id", "i", "j", "mutual_1p5_fF", "mutual_3_fF", "source"],
    )
    mi = _read("model_input/model_input-*.parquet", ["design_id", "X", "E", "G", "pri", "sig"])
    sch = json.load(open(SF / "schema.json"))
    nk, ek, gk = sch["nodekeys"], sch["edgekeys"], sch["globkeys"]
    nd = {d: g for d, g in nodes.groupby("design_id")}
    ed = {d: g for d, g in edges.groupby("design_id")}
    gd = glob_.set_index("design_id")
    od = {d: g for d, g in opt.groupby("design_id")}
    md = mi.set_index("design_id")
    bad = {"nodes": 0, "edges": 0, "globals": 0, "optimized_prior": 0, "model_input": 0}
    n = 0
    for p in sorted(SF.glob("n*.npz")):
        z = np.load(p)
        z = {k: z[k] for k in z.files}  # decompress once per shard, not per row
        for r, key in enumerate(z["key"].tolist()):
            did = key_to_design_id(key)
            n += 1
            X = nd[did].sort_values("node_idx")[nk].to_numpy(np.float32)
            bad["nodes"] += not np.array_equal(X, z["X"][r], equal_nan=True)
            E = ed[did].sort_values(["i", "j"])[ek].to_numpy(np.float32)
            bad["edges"] += not np.array_equal(E, z["E"][r], equal_nan=True)
            bad["globals"] += not np.array_equal(gd.loc[did, gk].to_numpy(np.float32), z["G"][r])
            o = od[did].sort_values(["i", "j"])
            bad["optimized_prior"] += not np.array_equal(
                o[["mutual_1p5_fF", "mutual_3_fF"]].to_numpy(np.float32), z["pri"][r]
            )
            m = md.loc[did]
            bad["model_input"] += not (
                np.array_equal(np.asarray(m.X, np.float32), z["X"][r].reshape(-1), equal_nan=True)
                and np.array_equal(np.asarray(m.E, np.float32), z["E"][r].reshape(-1))
                and np.array_equal(np.asarray(m.pri, np.float32), z["pri"][r].reshape(-1))
                and np.array_equal(np.asarray(m.sig), z["sig"][r])
            )
    out = dict(shard_rows_checked=n, mismatches=bad)
    # coarse prior vs its sources
    co = _read("physics_prior/physics_prior-*.parquet", ["design_id", "i", "j", "mutual_fF"])
    cd = {d: g.sort_values(["i", "j"]).mutual_fF.to_numpy() for d, g in co.groupby("design_id")}
    cb, cn = 0, 0
    for f in sorted(glob.glob(str(DL / "multifid/levels_hpc/*.pkl"))):
        for key, r in pickle.load(open(f, "rb"))["records"].items():
            cn += 1
            cb += not np.array_equal(cd[key_to_design_id(key)], np.asarray(r["levels"][0.375]["mutual"], np.float64))
    recs = _x1_records()
    for key, r in recs.items():
        n_ = r["n"]
        iu = np.triu_indices(n_, 1)
        cn += 1
        cb += not np.array_equal(cd[key_to_design_id(key)], np.asarray(r["prior"]["frozen"][0.375], np.float64)[iu])
    out["coarse_rows_checked"], out["coarse_mismatches"] = cn, cb
    out["bit_equal"] = cb == 0 and not any(bad.values())
    return out


def check_labels_vs_sources():
    """Public + heldout labels vs the shard C (float32, train/val) and the X1 test labels (float64): max relative difference."""
    from processed import SF

    L = pd.concat(
        [
            _read("labels/labels-*.parquet", ["simulation_id", "design_id", "net_a", "net_b", "C_fF"]),
            _read(
                "labels/labels-test.parquet",
                ["simulation_id", "design_id", "net_a", "net_b", "C_fF"],
                root=HELD / "data",
            ),
        ]
    )
    L = L[L.simulation_id.str.endswith("#q3d")]
    lab = {(d, a, b): c for d, a, b, c in zip(L.design_id, L.net_a, L.net_b, L.C_fF)}
    pi = pickle.load(open(DL / "multifid/pool_index.pkl", "rb"))
    worst, n, nan_lab, miss = 0.0, 0, 0, 0
    for p in sorted(SF.glob("n*.npz")):
        z = np.load(p)
        z = {k: z[k] for k in z.files}  # decompress once per shard, not per row
        for r, key in enumerate(z["key"].tolist()):
            if z["split"][r] == "train" and key not in pi["src"]:
                continue  # test rows: checked against the X1 labels below
            nm = pi["src"][key]["names"]
            did = key_to_design_id(key)
            for i in range(len(nm)):
                for j in range(i, len(nm)):
                    v = lab.get((did, nm[i], nm[j]), lab.get((did, nm[j], nm[i])))
                    c = z["C"][r][i, j]
                    if v is None:
                        miss += 1
                        continue
                    if not np.isfinite(c):
                        nan_lab += 1
                        continue
                    n += 1
                    worst = max(worst, abs(np.float32(v) - c) / max(abs(c), 1e-6))
    tw, tn, tmiss = 0.0, 0, 0
    man = json.load(open(DL / "v4test/x1_manifest.json"))
    T = {c: pickle.load(open(DL / f"v4test/labels/{c}.pkl", "rb")) for c in {e["cohort"] for e in man}}
    for e in man:
        Y = np.asarray(T[e["cohort"]][e["key"]], float)
        did = key_to_design_id(e["key"])
        nm = e["names"]
        for i in range(len(nm)):
            for j in range(i, len(nm)):
                if not np.isfinite(Y[i, j]):
                    continue
                v = lab.get((did, nm[i], nm[j]), lab.get((did, nm[j], nm[i])))
                if v is None:
                    tmiss += 1
                    continue
                tn += 1
                tw = max(tw, abs(v - Y[i, j]) / max(abs(Y[i, j]), 1e-9))
    return dict(
        train_val_entries=n,
        train_val_max_rel_diff_vs_shard_float32=float(worst),
        train_val_entries_missing=miss,
        shard_nan_entries=nan_lab,
        test_entries=tn,
        test_max_rel_diff_vs_x1_labels=float(tw),
        test_entries_missing=tmiss,
    )


def qc_summary():
    S = _read(
        "simulations/simulations-*.parquet",
        [
            "simulation_id",
            "design_id",
            "role",
            "status",
            "split",
            "qc_symmetric",
            "qc_signs",
            "qc_closure",
            "qc_pd",
            "qc_convergence",
            "qc_ok",
        ],
    )
    out = {"status": S.status.value_counts().to_dict(), "qc_convergence": S.qc_convergence.value_counts().to_dict()}
    ok = S[S.status == "ok"]
    for c in ("qc_symmetric", "qc_signs", "qc_closure", "qc_pd", "qc_ok"):
        out[f"{c}_false"] = int((ok[c] == False).sum())  # noqa: E712
    ok = ok.assign(collection=ok.design_id.str.split("/").str[1])
    out["qc_ok_false_by_collection_split"] = {
        f"{c}|{s}": int(n) for (c, s), n in ok[~ok.qc_ok.astype(bool)].groupby(["collection", "split"]).size().items()
    }
    return out


def sizes():
    out = {}
    for root, tag in ((OUT, "public"), (HELD, "heldout")):
        for d in sorted(p for p in (root / "data").rglob("*") if p.is_dir()) + [root / "gds"]:
            if not d.exists():
                continue
            b = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            out[f"{tag}:{d.relative_to(root)}"] = b
        out[f"{tag}:TOTAL"] = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    for t in (
        "designs",
        "layouts",
        "simulations",
        "labels",
        "label_passes",
        "geometric_embedding",
        "physics_prior",
        "optimized_physics_prior",
        "model_input",
    ):
        rows = 0
        for f in glob.glob(str(PUB / t / "*.parquet")):
            rows += pq.ParquetFile(f).metadata.num_rows
        out[f"rows:{t}"] = rows
    return out
