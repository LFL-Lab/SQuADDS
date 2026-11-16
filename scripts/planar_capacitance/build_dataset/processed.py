"""Processed tables from the HPC-computed shard sources (never recomputed here):

- geometric_embedding (nodes 61 / edges 42 / globals 4): QCap all-data shards `DL/flagship_v4/shards/acc_v2m_sf`
  (TRAIN+VAL from the paper's P2d shards, 5,809 test rows from the X1 cached inputs); the 92 QC-excluded test layouts
  from `DL/v4test/x1_cached_inputs.pkl` (the same source the shard test rows were packed from).
- physics_prior (single solve p = 0.375, v1 frozen solver): TRAIN+VAL `DL/multifid/levels_hpc`, test
  `DL/v4test/preds` (`prior["frozen"][0.375]`).
- optimized_physics_prior (C^R = R(1.5, 3), corrected stack v2m + substrate far field): the sf shards; test layouts not in
  the shards with N <= 9 from `DL/v4test/preds` `prior["deployed"]` (v2m = sf for N <= 9, checked bit-equal on every N <= 9
  shard test row); N > 9 layouts not in the shards have no prior of this version (reported).
- model_input: the sf shards packed per layout (X, E, G, pri, sig), labels excluded.
"""

from __future__ import annotations

import collections
import glob
import json
import pickle

import numpy as np
import pandas as pd
from common import DL, OUT, SCH, WORK, code_sha, key_to_design_id, to_table, write_parquet

SF = DL / "flagship_v4/shards/acc_v2m_sf"
V1SH = DL / "flagship_v4/shards/acc_v1"
SF_CODE = (
    "priorv2.py:1a0bdcb71500c7f11e63238afe531c18caabd5c9f6f7aaa73d9b7dcd0ad61f8c;"
    "farfield.py:6c877b47c2894fe78a38ad4199b29e450b610c2c45a261c8cf4d53efc1912836;"
    "fastprior_pkg.py:956f2b62c308ec1561bbc033716b52c24dc63a916b9373cc44090eb6514e5dde"
)
V2M_CODE = (
    SF_CODE.split(";farfield")[0] + ";fastprior_pkg.py:956f2b62c308ec1561bbc033716b52c24dc63a916b9373cc44090eb6514e5dde"
)
MF_CODE = (
    "md5 fastprior.py:21c4657827a5149cb230299a7642724a;fastprior_loc.py:037fc0cd601b444ca8ac366da59248ed;"
    "proxy.py:7f397c99b16523044392467b7b5e53c4;compute_levels.py (tl-squadds scripts/multifid)"
)


def _names():
    pi = pickle.load(open(DL / "multifid/pool_index.pkl", "rb"))
    out = {k: list(v["names"]) for k, v in pi["src"].items()}
    man = json.load(open(DL / "v4test/x1_manifest.json"))
    for e in man:
        out[e["key"]] = list(e["names"])
    return out, {e["key"]: e for e in man}


def _x1_records():
    recs = {}
    for f in sorted(glob.glob(str(DL / "v4test/preds/task_*.pkl"))):
        d = pickle.load(open(f, "rb"))
        for (_c, k), r in d["records"].items():
            recs[k] = dict(
                r,
                machine=f"{r.get('host')} ({r.get('cpu')}), {d['meta'].get('threads')} threads",
                x1_hashes=d["meta"]["hashes"],
            )
    return recs


def build(split_of: dict):
    csha = code_sha()
    names, man = _names()
    sch = json.load(open(SF / "schema.json"))
    nk, ek, gk = sch["nodekeys"], sch["edgekeys"], sch["globkeys"]
    assert tuple(nk) == SCH.NODE_FEATURES and tuple(ek) == SCH.EDGE_FEATURES and tuple(gk) == SCH.GLOBAL_FEATURES
    emb_sha = f"base_sha:{sch['base_sha']};extra_sha:{sch['extra_sha']}"
    nodes, edges, globs, opt, mi = [], [], [], [], []
    seen, v1_val_check = set(), []
    test_sf_vs_dep = []
    recs = _x1_records()
    for p in sorted(SF.glob("n*.npz")):
        z = np.load(p, allow_pickle=False)
        zv1 = np.load(V1SH / p.name, allow_pickle=False)
        # decompress each array once per shard (indexing the NpzFile per row re-read the array; row views pinned a full copy each)
        z, zv1 = {k: z[k] for k in z.files}, {k: zv1[k] for k in zv1.files}
        assert np.array_equal(z["key"], zv1["key"])
        n = int(p.stem[1:])
        pairs = z["pairs"]
        src = f"flagship_v4/shards/acc_v2m_sf/{p.name}"
        for r, key in enumerate(z["key"].tolist()):
            did = key_to_design_id(key)
            nm = names[key]
            assert len(nm) == n and nm[0] == "G", key
            seen.add(key)
            sp = split_of[did]
            X, E, G = z["X"][r], z["E"][r], z["G"][r]
            assert bool(X[0, nk.index("role_ground")] == 1) and not X[1:, nk.index("role_ground")].any(), key
            nodes.append((did, n, nm, X, sp, src))
            edges.append((did, nm, pairs, E, sp, src))
            globs.append((did, n, G, sp, src))
            pri = z["pri"][r]
            opt.append((did, nm, pairs, pri[:, 0], pri[:, 1], sp, src, SCH.OPTIMIZED_PRIOR_VERSION, SF_CODE))
            mi.append(
                dict(
                    design_id=did,
                    n_conductors=n,
                    net_order=nm,
                    model_family=str(z["family"][r]),
                    X=X.reshape(-1),
                    E=E.reshape(-1),
                    G=G,
                    pri=pri.reshape(-1),
                    sig=z["sig"][r],
                    embedding_version=SCH.EMBEDDING_VERSION,
                    prior_version=SCH.OPTIMIZED_PRIOR_VERSION,
                    source=src,
                    split=sp,
                )
            )
            if sp == "test" and n <= 9:
                dep = recs[key]["prior"]["deployed"]
                iu = (pairs[:, 0], pairs[:, 1])
                test_sf_vs_dep.append(
                    bool(
                        np.array_equal(
                            np.stack([np.asarray(dep[1.5])[iu], np.asarray(dep[3.0])[iu]], -1).astype(np.float32), pri
                        )
                    )
                )
            if sp == "val":
                v1_val_check.append((key, zv1["pri"][r]))
    print(
        "shard rows",
        len(seen),
        "test N<=9 sf == deployed v2m (float32):",
        sum(test_sf_vs_dep),
        "/",
        len(test_sf_vs_dep),
        flush=True,
    )
    sf_eq_dep_n9 = all(test_sf_vs_dep)
    # test layouts not in the shards: embedding from the cached inputs, optimized prior only where N <= 9
    cached = pickle.load(open(DL / "v4test/x1_cached_inputs.pkl", "rb"))
    missing = collections.Counter()
    missing_list = []
    wi = nk.index("log_min_halfwidth")
    for key, e in man.items():
        if key in seen:
            continue
        did = key_to_design_id(key)
        sp = split_of[did]
        nm = e["names"]
        n = len(nm)
        ck = (e["cohort"], key)
        src = "v4test/x1_cached_inputs.pkl"
        if ck in cached:
            cx = cached[ck]
            assert list(cx["names"]) == nm
            pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
            X = np.array([[nd[k] for k in nk[:-1]] for nd in cx["nodes"]], np.float32)
            X = np.concatenate([X, (~np.isfinite(X[:, wi])).astype(np.float32)[:, None]], -1)
            E = np.array([[cx["edges"][tuple(ij)][k] for k in ek] for ij in pairs], np.float32)
            G = np.array([cx["global_features"][k] for k in gk], np.float32)
            nodes.append((did, n, nm, X, sp, src))
            edges.append((did, nm, pairs, E, sp, src))
            globs.append((did, n, G, sp, src))
        else:
            missing["embedding"] += 1
            missing_list.append(f"{did}: no embedding (not in the X1 cached inputs)")
        if n <= 9 and sf_eq_dep_n9 and key in recs:
            dep = recs[key]["prior"]["deployed"]
            pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
            iu = (pairs[:, 0], pairs[:, 1])
            opt.append(
                (
                    did,
                    nm,
                    pairs,
                    np.asarray(dep[1.5])[iu].astype(np.float32),
                    np.asarray(dep[3.0])[iu].astype(np.float32),
                    sp,
                    "v4test/preds prior['deployed'] (v2m; = sf for N <= 9, verified)",
                    SCH.OPTIMIZED_PRIOR_VERSION,
                    V2M_CODE,
                )
            )
        else:
            missing["optimized_prior"] += 1
            missing_list.append(
                f"{did}: no optimized prior of version {SCH.OPTIMIZED_PRIOR_VERSION} (N = {n}; not in the sf shards)"
            )
    print("not in shards:", dict(missing), flush=True)

    # write embedding tables
    _write_nodes(nodes, nk, emb_sha, csha)
    _write_edges(edges, ek, emb_sha, csha)
    g = pd.DataFrame(
        [
            dict(design_id=d, n_conductors=n, **{k: G[i] for i, k in enumerate(gk)}, split=sp, source=s)
            for d, n, G, sp, s in globs
        ]
    )
    g = g.assign(embedding_version=SCH.EMBEDDING_VERSION, embedding_code_sha256=emb_sha, build_code_sha256=csha)
    for sp, x in g.groupby("split"):
        write_parquet(
            to_table(x, "geometric_embedding_globals"), OUT / f"data/geometric_embedding/globals-{sp}.parquet", False
        )
    # optimized prior
    rows = []
    for did, nm, pairs, c15, c3, sp, src, ver, code in opt:
        nm = np.array(nm)
        r = np.exp(2 * np.log(c3.astype(np.float64)) - np.log(c15.astype(np.float64)))
        rows.append(
            pd.DataFrame(
                dict(
                    design_id=did,
                    net_a=nm[pairs[:, 0]],
                    net_b=nm[pairs[:, 1]],
                    i=pairs[:, 0].astype(np.int16),
                    j=pairs[:, 1].astype(np.int16),
                    mutual_1p5_fF=c15,
                    mutual_3_fF=c3,
                    mutual_R_fF=r,
                    C_R_fF=-r,
                    prior_version=ver,
                    prior_code_sha256=code,
                    source=src,
                    split=sp,
                )
            )
        )
    o = pd.concat(rows, ignore_index=True).assign(build_code_sha256=csha)
    for sp, x in o.groupby("split"):
        write_parquet(
            to_table(x, "optimized_physics_prior"),
            OUT / f"data/optimized_physics_prior/optimized_physics_prior-{sp}.parquet",
            True,
            meta={"squadds.prior_version": SCH.OPTIMIZED_PRIOR_VERSION},
        )
    n_opt = o.design_id.nunique()
    del o, rows
    # coarse prior
    n_coarse, cmiss = _coarse(split_of, names, man, recs, csha, v1_val_check)
    missing_list += cmiss
    # model input
    m = pd.DataFrame(mi).assign(build_code_sha256=csha)
    for sp, x in m.groupby("split"):
        write_parquet(
            to_table(x, "model_input"),
            OUT / f"data/model_input/model_input-{sp}.parquet",
            False,
            meta={
                "squadds.prior_version": SCH.OPTIMIZED_PRIOR_VERSION,
                "squadds.embedding_version": SCH.EMBEDDING_VERSION,
                "squadds.pairs": "upper triangle (i < j) of net_order, row-major",
                "squadds.X_shape": "N x 61",
                "squadds.E_shape": "P x 42",
                "squadds.pri_shape": "P x 2 (1.5, 3 ppg)",
            },
        )
    summ = dict(
        shard_rows=len(seen),
        embedding_layouts=len(globs),
        optimized_prior_layouts=n_opt,
        coarse_prior_layouts=n_coarse,
        model_input_rows=len(m),
        test_n9_sf_equals_deployed=[int(sum(test_sf_vs_dep)), len(test_sf_vs_dep)],
        missing=missing_list,
    )
    json.dump(summ, open(WORK / "processed_summary.json", "w"), indent=1)
    return summ


def _write_nodes(nodes, nk, emb_sha, csha):
    by = collections.defaultdict(list)
    for did, n, nm, X, sp, src in nodes:
        df = pd.DataFrame(X, columns=nk)
        df.insert(0, "design_id", did)
        df.insert(1, "node_idx", np.arange(n, dtype=np.int16))
        df.insert(2, "net", nm)
        df["source"], df["split"] = src, sp
        by[sp].append(df)
    for sp, parts in by.items():
        x = pd.concat(parts, ignore_index=True).assign(
            embedding_version=SCH.EMBEDDING_VERSION, embedding_code_sha256=emb_sha, build_code_sha256=csha
        )
        write_parquet(
            to_table(x, "geometric_embedding_nodes"), OUT / f"data/geometric_embedding/nodes-{sp}.parquet", True
        )


def _write_edges(edges, ek, emb_sha, csha):
    by = collections.defaultdict(list)
    for did, nm, pairs, E, sp, src in edges:
        nm = np.array(nm)
        df = pd.DataFrame(E, columns=ek)
        df.insert(0, "design_id", did)
        df.insert(1, "i", pairs[:, 0].astype(np.int16))
        df.insert(2, "j", pairs[:, 1].astype(np.int16))
        df.insert(3, "net_a", nm[pairs[:, 0]])
        df.insert(4, "net_b", nm[pairs[:, 1]])
        df["source"], df["split"] = src, sp
        by[sp].append(df)
    for sp, parts in by.items():
        x = pd.concat(parts, ignore_index=True).assign(
            embedding_version=SCH.EMBEDDING_VERSION, embedding_code_sha256=emb_sha, build_code_sha256=csha
        )
        write_parquet(
            to_table(x, "geometric_embedding_edges"), OUT / f"data/geometric_embedding/edges-{sp}.parquet", True
        )


def _coarse(split_of, names, man, recs, csha, v1_val_check):
    rows, miss, seen = [], [], set()
    val_eq = []
    v1v = dict(v1_val_check)
    for f in sorted(glob.glob(str(DL / "multifid/levels_hpc/*.pkl"))):
        d = pickle.load(open(f, "rb"))
        mach = f"{d['meta']['host']} ({d['meta']['cpu']}), {d['meta']['threads']} threads"
        for key, r in d["records"].items():
            nm = np.array(names[key])
            n = len(nm)
            assert r["n"] == n
            iu = np.triu_indices(n, 1)
            m = np.asarray(r["levels"][0.375]["mutual"], np.float64)
            did = key_to_design_id(key)
            rows.append(
                pd.DataFrame(
                    dict(
                        design_id=did,
                        net_a=nm[iu[0]],
                        net_b=nm[iu[1]],
                        i=iu[0].astype(np.int16),
                        j=iu[1].astype(np.int16),
                        mutual_fF=m,
                        C_fF=-m,
                        machine=mach,
                        source=f"multifid/levels_hpc/{f.rsplit('/', 1)[-1]}",
                        prior_code_sha256=MF_CODE,
                        split=split_of[did],
                    )
                )
            )
            seen.add(key)
            if key in v1v and 1.5 in r["levels"]:
                pri = v1v[key]
                val_eq.append(
                    bool(
                        np.array_equal(
                            np.stack(
                                [np.asarray(r["levels"][1.5]["mutual"]), np.asarray(r["levels"][3.0]["mutual"])], -1
                            ).astype(np.float32),
                            pri,
                        )
                    )
                )
    for key, e in man.items():
        did = key_to_design_id(key)
        if key not in recs:
            miss.append(f"{did}: no coarse prior (no X1 record)")
            continue
        r = recs[key]
        nm = np.array(e["names"])
        n = len(nm)
        iu = np.triu_indices(n, 1)
        m = np.asarray(r["prior"]["frozen"][0.375], np.float64)[iu]
        rows.append(
            pd.DataFrame(
                dict(
                    design_id=did,
                    net_a=nm[iu[0]],
                    net_b=nm[iu[1]],
                    i=iu[0].astype(np.int16),
                    j=iu[1].astype(np.int16),
                    mutual_fF=m,
                    C_fF=-m,
                    machine=r["machine"],
                    source="v4test/preds prior['frozen'][0.375]",
                    prior_code_sha256=f"fastprior_pkg.py:{r['x1_hashes']['fastprior_pkg.py']} (frozen v1 mode)",
                    split=split_of[did],
                )
            )
        )
        seen.add(key)
    x = pd.concat(rows, ignore_index=True).assign(
        ppg=np.float32(SCH.COARSE_PPG), prior_version=SCH.COARSE_PRIOR_VERSION, build_code_sha256=csha
    )
    for sp, g in x.groupby("split"):
        write_parquet(
            to_table(g, "physics_prior"),
            OUT / f"data/physics_prior/physics_prior-{sp}.parquet",
            True,
            meta={"squadds.prior_version": SCH.COARSE_PRIOR_VERSION},
        )
    print(
        f"coarse prior layouts {len(seen)}; VAL v1 levels (1.5, 3) == acc_v1 shard pri: {sum(val_eq)} / {len(val_eq)}",
        flush=True,
    )
    json.dump(dict(val_levels_equal_acc_v1=[int(sum(val_eq)), len(val_eq)]), open(WORK / "coarse_check.json", "w"))
    return len(seen), miss
