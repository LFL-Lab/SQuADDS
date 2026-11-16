"""Raw tables: designs, layouts (+ GDS copies), simulations, labels, label_passes.

v1 (local build `DL/release_core`) rows are carried over unchanged (keys, GDS
bytes, label values); new collections (two-net, mixed-shape synthetic Q3D, campaign 450) are read from their raw Q3D run
records. Every simulation's QC flags are recomputed. Splits follow the paper (P2d shards: train/val; X1 manifest: test).
"""

from __future__ import annotations

import glob
import hashlib
import json
import pickle
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from common import (
    C_HPC,
    DL,
    HELD,
    HF_MAX_FILES_PER_DIR,
    HPC_RAW,
    OUT,
    PROV,
    PUBLIC_TEST_COLLECTIONS,
    PUBLIC_TEST_REASON,
    SCH,
    TRAIN_STACK,
    V1,
    WORK,
    entry_kind,
    gds_info,
    gds_relpath,
    jmin,
    key_to_design_id,
    layer_map,
    qc_convergence,
    qc_flags,
    read_conv,
    read_csv_passes,
    sha256_file,
    to_table,
    write_parquet,
)

TABLE_I = {  # collection -> Table I row (paper lingo)
    "exp6": "interdigital",
    "exp7": "interdigital",
    "exp8": "interdigital",
    "exp9": "paddle",
    "exp10": "concentric",
    "exp11": "star",
    "exp12": "multipad star",
    "exp13": "Xmon",
    "exp14": "TransmonPocket",
    "exp15": "shielded concentric",
    "twonet": "single conductor in a ground cut",
    "random_v1": "random shape placements",
    "synth_q3d": "mixed-shape layouts",
    "wonky_v1": "freeform geometries",
    "atlas": "all quantum-metal QComponents",
    "largen": "large N",
    "qm116": "unseen qubits and couplers",
    "p3rand36": "held-out TransmonPockets",
}
B_ROW = {"B1": "local edge effects", "B2": "third-body couplings", "B3": "crowded pockets"}
GRID_ROW = {
    "grid_qubit_1": "one-qubit lattice",
    "grid_qubit_5": "five-qubit lattice",
    "grid_qubit_9": "nine-qubit lattice",
}
COMPONENT = {
    "exp6": "GeneralizedCapNInterdigital",
    "exp7": "GeneralizedCapNInterdigital",
    "exp8": "GeneralizedCapNInterdigital",
    "exp9": "GeneralizedCapNInterdigital",
    "exp10": "GeneralizedCapConcentric",
    "exp11": "GeneralizedCapStar",
    "exp12": "GeneralizedCapStar",
    "exp13": "Xmon",
    "exp14": "TransmonPocket",
    "exp15": "ShieldedCapConcentric",
}
NEW = ("twonet", "synth_q3d", "campaign450")


# ----------------------------------------------------------------------------------------------- split sources
def paper_split():
    """design_id -> (split, model_family, names) from the P2d pool (train/val) and the X1 manifest (test)."""
    pi = pickle.load(open(DL / "multifid/pool_index.pkl", "rb"))
    out = {}
    for k, v in pi["pool"].items():
        out[key_to_design_id(k)] = (v["split"], v["family"], list(pi["src"][k]["names"]), k, None)
    for e in json.load(open(DL / "v4test/x1_manifest.json")):
        d = key_to_design_id(e["key"])
        assert d not in out, d
        out[d] = ("test", None, list(e["names"]), e["key"], e["cohort"])
    return out


def declared_test_partition() -> set[str]:
    """Designs in a declared test partition that are not in X1 (excluded by QC): their labels stay private too."""
    s = set()
    q = pd.read_parquet(DL / "diverse/hpc_copy/twonet/qc.parquet", columns=["id", "split"])
    s |= {f"ref/twonet/q3d_cap_{i}" for i in q.loc[q.split == "test", "id"]}
    sel = pd.read_csv(DL / "synth_q3d/selection.csv", dtype={"id": str})
    for i, ci in zip(sel["id"], sel["i"]):
        if int(hashlib.sha256(f"corpus/{int(ci):06d}".encode()).hexdigest(), 16) % 100 >= 90:
            s.add(f"ref/synth_q3d/q3d_cap_{i}")
    return s


# ----------------------------------------------------------------------------------------------- new collections
def _run_worker(args):
    did, run, prov = args
    run = Path(run)
    res = json.load(open(run / "result.json")) if (run / "result.json").exists() else {}
    conv = read_conv(next(iter(glob.glob(str(run / "*.conv"))), None))
    out = dict(
        design_id=did,
        run=str(run),
        prov=prov,
        res=res,
        conv=conv,
        done=(run / "DONE").exists(),
        truncated=(run / "TRUNCATED").exists(),
        error=(run / "error.txt").exists(),
    )
    if (run / "cap_matrix.csv").exists():
        passes, nets, C, freq = read_csv_passes(run / "cap_matrix.csv")
        if len(passes):
            out.update(passes=passes, nets=nets, C=C, freq=freq, csv_sha=sha256_file(run / "cap_matrix.csv"))
    return out


def new_designs(ps):
    """Design rows + run jobs for two-net, synthetic Q3D and campaign 450."""
    rows, jobs = [], []
    tw = {m.get("id", f"{i:04d}"): m for i, m in enumerate(json.load(open(DL / "twonet/designs/_manifest.json")))}
    sel = pd.read_csv(DL / "synth_q3d/selection.csv", dtype={"id": str}).set_index("id")
    camp = pd.read_csv(DL / "campaign_450/designs.csv", dtype={"id": str}).set_index("id")
    for coll, root, gen in (
        ("twonet", DL / "twonet/designs", "tl-squadds scripts/q3d_campaign.py twonet (PARAM2_TWONET)"),
        ("synth_q3d", DL / "synth_q3d/designs", "tl-squadds scripts/escn/gen.py escn_gen_v0 (seed 20261015)"),
        ("campaign450", DL / "campaign_450/designs", "tl-squadds PROTOCOL_DATA_CAMPAIGN (campaign 450)"),
    ):
        for dd in sorted(root.glob("q3d_cap_*")):
            i = dd.name.split("_")[-1]
            did = f"ref/{coll}/{dd.name}"
            js = json.load(open(dd / f"cap_{i}.json")) if (dd / f"cap_{i}.json").exists() else {}
            params = {k: v for k, v in js.items() if k not in ("ansys_gds", "mesh", "layer_stack")}
            fam, row, seed = coll, None, None
            if coll == "twonet":
                m = tw.get(i, {})
                params["_manifest"] = {k: v for k, v in m.items() if k != "nets"}
                fam, row = "single_conductor_ground_cut", TABLE_I["twonet"]
            elif coll == "synth_q3d":
                params["_selection"] = sel.loc[i].to_dict() if i in sel.index else None
                fam, row = "mixed_shape", TABLE_I["synth_q3d"]
            else:
                c = camp.loc[i].to_dict() if i in camp.index else {}
                params["_campaign"] = c
                b = c.get("bucket")
                fam, row = f"campaign450_{b}", B_ROW.get(b)
                seed = c.get("seed")
            rows.append(
                dict(
                    design_id=did,
                    collection=coll,
                    family=fam,
                    paper_family=row,
                    gds_src=str(dd / f"cap_{i}.gds"),
                    params=jmin(params),
                    contributor_id="sadman_shanto",
                    generator=gen,
                    generator_seed=pd.to_numeric(seed, errors="coerce"),
                    group=f"{coll}|{dd.name}",
                    notes=None,
                    net_order=None,
                    split_v1=None,
                    setup=js,
                )
            )
            run = HPC_RAW / coll.replace("synth_q3d", "synth") / "runs" / dd.name
            jobs.append((did, str(run), f"{C_HPC}/{coll.replace('synth_q3d', 'synth')}/runs/{dd.name}"))
    return rows, jobs


# ----------------------------------------------------------------------------------------------- build
def build(workers: int):
    WORK.mkdir(parents=True, exist_ok=True)
    ps = paper_split()
    dtest = declared_test_partition()
    v1d = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(V1 / "data/v1/designs/*.parquet")))], ignore_index=True
    )
    v1l = pd.read_parquet(V1 / "data/v1/layouts/layouts.parquet")
    v1s = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(V1 / "data/v1/simulations/*.parquet")))], ignore_index=True
    )
    print("v1", len(v1d), "designs", len(v1l), "layouts", len(v1s), "simulations", flush=True)

    # designs
    D = v1d.rename(columns={"split": "split_v1"}).copy()
    D["contributor_id"] = D["contributor_id"].replace(
        {"grid_collaborator": "sadman_shanto"}
    )  # lattices: author 2026-10-30
    D["gds_src"] = [str(V1 / "gds" / f"{d}.gds") for d in D.design_id]
    nrows, jobs = new_designs(ps)
    N = pd.DataFrame(nrows)
    setups = {r["design_id"]: r.pop("setup") for r in nrows}
    D = pd.concat([D, N.drop(columns="setup")], ignore_index=True)
    assert not D.design_id.duplicated().any()
    D["paper_family"] = [
        r if isinstance(r, str) else TABLE_I.get(c) for r, c in zip(D.get("paper_family"), D.collection)
    ]
    D.loc[D.collection == "grid", "paper_family"] = [
        GRID_ROW[d.split("/")[-1]] for d in D.loc[D.collection == "grid", "design_id"]
    ]
    sp, mf = [], []
    for d, s1, coll in zip(D.design_id, D.split_v1, D.collection):
        if d in ps:
            sp.append(ps[d][0])
            mf.append(ps[d][1])
        elif s1 == "test" or d in dtest:
            sp.append("test")
            mf.append(None)
        elif coll in TABLE_I or coll == "campaign450":
            sp.append("excluded" if (s1 in ("train", "val") or coll in NEW) else (s1 or "none"))
            mf.append(None)
        else:
            sp.append(s1 or "none")
            mf.append(None)
    D["split"], D["model_family"] = sp, mf
    test_family = {
        d: (
            "external"
            if v[4] in ("atlas", "qm116", "p3rand36", "unusual", "largen") or (v[4] or "").startswith("grid")
            else None
        )
        for d, v in ps.items()
        if v[0] == "test"
    }
    D["model_family"] = [m if m else test_family.get(d) for d, m in zip(D.design_id, D.model_family)]
    cf = D[D.split.isin(["train", "val"])].groupby("collection").model_family.agg(lambda x: x.mode().iat[0]).to_dict()
    D["model_family"] = [
        m
        if m
        else (("synth_q3d_N<=7" if len(ps[d][2]) <= 7 else "synth_q3d_N>=8") if c == "synth_q3d" else cf.get(c))
        if d in ps
        else m
        for d, m, c in zip(D.design_id, D.model_family, D.collection)
    ]
    D["component_class"] = D.collection.map(COMPONENT)
    D["squadds_config"] = D.collection.map(SCH.squadds_config)
    ov = WORK / "overlap_map.parquet"
    ref = pd.read_parquet(ov).set_index("design_id")["squadds_db_ref"].to_dict() if ov.exists() else {}
    D["squadds_db_ref"] = D.design_id.map(ref)
    print(
        "overlap refs attached:",
        int(D.squadds_db_ref.notna().sum()),
        "(overlap_map present)" if ov.exists() else "(no overlap map yet)",
        flush=True,
    )

    # simulations of the new collections
    split_now = dict(zip(D.design_id, D.split))
    sims, lab_parts, pass_parts, new_order = [], [], [], {}
    with Pool(workers) as pool:
        recs = list(pool.imap_unordered(_run_worker, jobs, chunksize=16))
    names_of = {d: v[2] for d, v in ps.items()}
    for o in sorted(recs, key=lambda r: r["design_id"]):
        did = o["design_id"]
        js = setups[did]
        res, conv = o["res"], o["conv"]
        setup = {
            "ansys_gds": js.get("ansys_gds"),
            "mesh": js.get("mesh"),
            "layer_stack": js.get("layer_stack"),
            "overrides": res.get("setup_overrides"),
        }
        status = (
            "missing"
            if not Path(o["run"]).exists()
            else ("failed_no_pass" if "C" not in o else ("incomplete" if o["truncated"] or not o["done"] else "ok"))
        )
        sim = dict(
            simulation_id=f"{did}#q3d",
            design_id=did,
            sim_tag="q3d",
            solver="Ansys Q3D Extractor",
            solver_version=None,
            role="reference",
            **TRAIN_STACK,
            metal_model="thin_sheet",
            metal_t_um=None,
            freq_GHz=o.get("freq", 13.0),
            setup=jmin(setup),
            status=status,
            passes_completed=conv.get("passes_completed", int(o["passes"][-1]) if "C" in o else None),
            converged=conv.get("converged"),
            delta_pct_final=conv.get("delta_pct_final"),
            delta_target_pct=conv.get("delta_target_pct"),
            pass_history=jmin(conv.get("pass_history")) if conv else None,
            cores=res.get("cores"),
            solve_s=res.get("solve_s"),
            wall_s=res.get("wall_s"),
            peak_mem_GB=None,
            machine="USC CARC Discovery/Endeavour (cluster per job ledger)",
            provenance=jmin(
                {"path": o["prov"] + "/cap_matrix.csv", "sha256_csv": o.get("csv_sha"), "local_copy": o["run"]}
            ),
            contributor_id="sadman_shanto",
        )
        if "C" in o:
            nets = o["nets"]
            order = names_of.get(did) or (["G"] + [n for n in nets if n != "G"])
            assert sorted(order) == sorted(nets), (did, order, nets)
            perm = [nets.index(n) for n in order]
            C = o["C"][:, perm][:, :, perm]
            sim.update(qc_flags(C[-1], order, C[-2] if len(C) > 1 else None))
            new_order[did] = order
            iu = np.triu_indices(len(order))
            k = len(iu[0])
            lab_parts.append(
                pd.DataFrame(
                    dict(
                        simulation_id=sim["simulation_id"],
                        design_id=did,
                        net_a=np.array(order)[iu[0]],
                        net_b=np.array(order)[iu[1]],
                        a_idx=iu[0].astype(np.int16),
                        b_idx=iu[1].astype(np.int16),
                        C_fF=C[-1][iu],
                        contributor_id="sadman_shanto",
                    )
                )
            )
            P = len(o["passes"])
            pass_parts.append(
                pa.table(
                    dict(
                        simulation_id=pa.array(np.full(P * k, sim["simulation_id"])),
                        design_id=pa.array(np.full(P * k, did)),
                        net_a=pa.array(np.tile(np.array(order)[iu[0]], P)),
                        net_b=pa.array(np.tile(np.array(order)[iu[1]], P)),
                        a_idx=np.tile(iu[0], P).astype(np.int16),
                        b_idx=np.tile(iu[1], P).astype(np.int16),
                        **{"pass": np.repeat(o["passes"], k).astype(np.int16)},
                        C_fF=C[:, iu[0], iu[1]].reshape(-1),
                        split=pa.array(np.full(P * k, split_now[did])),
                    )
                )
            )
        sim["qc_convergence"] = qc_convergence(sim)
        sims.append(sim)
    D["net_order"] = [
        new_order.get(d, o if o is not None else names_of.get(d)) for d, o in zip(D.design_id, D.net_order)
    ]
    S_new = pd.DataFrame(sims)
    print("new simulations", S_new.status.value_counts().to_dict(), flush=True)

    # v1 simulations: carry everything, recompute QC from the v1 label tables (+ the raw symmetry recorded by v1)
    S1 = v1s.copy()
    S1["pass_history"] = [jmin(list(h)) if h is not None else None for h in S1.pass_history]
    S1["provenance"] = [jmin(p) for p in S1.provenance]
    sym = {s: (q or {}).get("sym_rel") for s, q in zip(S1.simulation_id, S1.qc)}
    S1 = S1.drop(columns=["qc", "cores_profile", "source", "licence_id", "dataset_version_added"])
    lab1 = pd.concat(
        [
            pd.read_parquet(f)
            for f in sorted(glob.glob(str(V1 / "data/v1/labels/*.parquet")))
            + [str(V1 / "private_heldout/data/v1/labels/labels-test.parquet")]
        ],
        ignore_index=True,
    )
    pas1 = pa.concat_tables(
        [
            pq.read_table(f, columns=["simulation_id", "design_id", "net_a", "net_b", "a_idx", "b_idx", "pass", "C_fF"])
            for f in sorted(glob.glob(str(V1 / "data/v1/label_passes/*.parquet")))
            + [str(V1 / "private_heldout/data/v1/label_passes/label_passes-test.parquet")]
        ]
    )
    prev = _prev_pass(pas1)
    qrows = []
    order_of = dict(zip(D.design_id, D.net_order))
    for sid, g in lab1.groupby("simulation_id", sort=False):
        did = g.design_id.iat[0]
        order = list(order_of[did])
        ix = {n: i for i, n in enumerate(order)}
        n = len(order)
        C = np.full((n, n), np.nan)
        a, b = g.net_a.map(ix).to_numpy(), g.net_b.map(ix).to_numpy()
        C[a, b] = g.C_fF.to_numpy()
        C[b, a] = g.C_fF.to_numpy()
        Cp = None
        if sid in prev:
            pa_, pb_, pv = prev[sid]
            Cp = np.full((n, n), np.nan)
            ia, ib = np.array([ix[x] for x in pa_]), np.array([ix[x] for x in pb_])
            Cp[ia, ib] = pv
            Cp[ib, ia] = pv
        q = qc_flags(C, order, Cp, sym_rel=sym.get(sid))
        q["simulation_id"] = sid
        qrows.append(q)
    S1 = S1.merge(pd.DataFrame(qrows), on="simulation_id", how="left")
    for c in ("qc_symmetric", "qc_signs", "qc_closure", "qc_pd", "converged"):
        S1[c] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else bool(v) for v in S1[c]]
    S1["qc_convergence"] = [qc_convergence(r) for r in S1.to_dict("records")]
    S = pd.concat([S1, S_new], ignore_index=True)
    S["qc_ok"] = [
        bool(
            r["status"] == "ok"
            and r.get("qc_symmetric")
            and r.get("qc_signs")
            and r.get("qc_closure") is not False
            and r.get("qc_pd") is not False
            and r["qc_convergence"] in ("converged", "unrecorded")
        )
        for r in S.to_dict("records")
    ]
    assert not S.simulation_id.duplicated().any()
    split_of = dict(zip(D.design_id, D.split))
    S["split"] = S.design_id.map(split_of)

    # labels: v1 rows (unchanged values) + new
    labN = pd.concat(lab_parts, ignore_index=True)
    labN["mutual_fF"] = np.where(labN.net_a != labN.net_b, -labN.C_fF, np.nan)
    labN["kind"] = [entry_kind(a, b) for a, b in zip(labN.net_a, labN.net_b)]
    labN["scored"] = ~(
        ((labN.net_a == labN.net_b) & (labN.net_a == "G")) | ((labN.net_a != labN.net_b) & (labN.C_fF == 0))
    )
    labN["dist_um"] = np.nan
    lab1 = lab1.drop(columns=["source", "licence_id", "dataset_version_added", "split"])
    L = pd.concat([lab1, labN], ignore_index=True)
    ok = set(S.loc[S.status == "ok", "simulation_id"])
    L = L[L.simulation_id.isin(ok)]
    L["split"] = L.design_id.map(split_of)
    passN = pa.concat_tables(pass_parts)
    passN = passN.filter(pc.is_in(passN.column("simulation_id"), pa.array(sorted(ok))))

    # GDS copies + layouts
    jobs = [
        (g, str(OUT / gds_relpath(d)))
        for d, g in zip(D.design_id, D.gds_src)
        if isinstance(g, str) and Path(g).exists()
    ]
    per_dir = pd.Series([str(Path(d).parent) for _, d in jobs]).value_counts()
    assert per_dir.max() <= HF_MAX_FILES_PER_DIR, per_dir.head()
    known = dict(zip(v1l.layout_id, v1l.to_dict("records")))
    with Pool(workers) as pool:
        info = dict(pool.imap_unordered(_copy_gds, [(s, d, None) for s, d in jobs], chunksize=64))
    lay, drow = {}, {}
    for d in D.design_id:
        dst = str(OUT / gds_relpath(d))
        if dst not in info:
            continue
        fs = info[dst]["sha256"]
        lid = "sha256:" + fs
        if lid in known:
            k = known[lid]
            rec = dict(
                layout_id=lid,
                geom_hash=k["geom_hash"],
                sha256=fs,
                unit_m=k["unit_m"],
                precision_m=k["precision_m"],
                top_cell=k["top_cell"],
                layer_map=k["layer_map"],
                n_polygons=k["n_polygons"],
                n_vertices=k["n_vertices"],
                bbox_um=list(k["bbox_um"]),
                file_bytes=k["file_bytes"],
                v1_path=k["path"],
                n_cond_layout=None,
            )
        else:
            gi = gds_info(dst)
            rec = dict(
                layout_id=lid,
                geom_hash=gi["geom_hash"],
                sha256=fs,
                unit_m=gi["unit_m"],
                precision_m=gi["precision_m"],
                top_cell=gi["top_cell"],
                layer_map=layer_map(gi["layers"]),
                n_polygons=gi["n_polygons"],
                n_vertices=gi["n_vertices"],
                bbox_um=gi["bbox_um"],
                file_bytes=gi["file_bytes"],
                v1_path=None,
                n_cond_layout=gi["n_cond_layout"],
            )
        rec["path"] = gds_relpath(d)
        lay.setdefault(lid, rec)
        drow[d] = rec
    D["layout_id"] = D.design_id.map(lambda d: drow[d]["layout_id"] if d in drow else None)
    D["geom_hash"] = D.design_id.map(lambda d: drow[d]["geom_hash"] if d in drow else None)
    newd = D.collection.isin(NEW)
    D.loc[newd, "chip_w_um"] = [
        drow[d]["bbox_um"][2] - drow[d]["bbox_um"][0] if d in drow else None for d in D.design_id[newd]
    ]
    D.loc[newd, "chip_h_um"] = [
        drow[d]["bbox_um"][3] - drow[d]["bbox_um"][1] if d in drow else None for d in D.design_id[newd]
    ]
    D["n_conductors"] = pd.array([len(o) if o is not None else None for o in D.net_order], dtype="Int16")
    D["generator_seed"] = pd.to_numeric(D.generator_seed, errors="coerce").astype("Int64")
    D["labelled"] = D.design_id.isin(set(L.design_id))
    LY = pd.DataFrame(list(lay.values()))
    nd = D.drop_duplicates("layout_id").set_index("layout_id")
    LY["n_conductors_layout"] = [
        n if n is not None else nd.n_conductors.get(lid) for n, lid in zip(LY.n_cond_layout, LY.layout_id)
    ]
    LY = LY.drop(columns="n_cond_layout").assign(contributor_id=LY.layout_id.map(nd.contributor_id), **PROV)
    LY["n_conductors_layout"] = pd.to_numeric(LY.n_conductors_layout, errors="coerce").astype("Int16")
    for c in ("n_polygons", "n_vertices", "file_bytes"):
        LY[c] = LY[c].astype("int32")
    print(f"layouts {len(LY)} (new {int(LY.v1_path.isna().sum())}); gds copies {len(info)}", flush=True)
    return D, LY, S, L, pas1, passN


def _prev_pass(t) -> dict:
    """simulation_id -> (net_a, net_b, C) of the second-to-last pass."""
    df = t.to_pandas()
    last = df.groupby("simulation_id")["pass"].transform("max")
    df = df[df["pass"] < last]
    if df.empty:
        return {}
    pm = df.groupby("simulation_id")["pass"].transform("max")
    df = df[df["pass"] == pm]
    return {s: (g.net_a.to_numpy(), g.net_b.to_numpy(), g.C_fF.to_numpy()) for s, g in df.groupby("simulation_id")}


def _copy_gds(args):
    src, dst, _ = args
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if not Path(dst).exists() or Path(dst).stat().st_size != Path(src).stat().st_size:
        shutil.copyfile(src, dst)
    s1 = sha256_file(dst)
    assert s1 == sha256_file(src), dst
    return dst, {"sha256": s1}


# ----------------------------------------------------------------------------------------------- write
def private(design_ids: pd.Series, private_ids: set) -> np.ndarray:
    return design_ids.isin(private_ids).to_numpy()


def write(D, LY, S, L, pas1, passN):
    pub_test = (D.split == "test") & D.collection.isin(PUBLIC_TEST_COLLECTIONS)
    hashes_test = set(D.loc[(D.split == "test") & ~pub_test, "geom_hash"].dropna())
    priv = set(D.loc[((D.split == "test") & ~pub_test) | D.geom_hash.isin(hashes_test), "design_id"])
    D["notes"] = [
        n
        if not (d in priv and s != "test")
        else ((n + "; ") if n else "") + "labels private: geometry equals a test design"
        for d, n, s in zip(D.design_id, D.notes, D.split)
    ]
    D.loc[pub_test, "notes"] = [((n + "; ") if n else "") + PUBLIC_TEST_REASON for n in D.loc[pub_test, "notes"]]
    D["source"], D["licence_id"], D["dataset_version_added"] = (
        PROV["source"],
        PROV["licence_id"],
        [
            "1.0.0-dev" if s is not None or c not in NEW else PROV["dataset_version_added"]
            for s, c in zip(D.split_v1, D.collection)
        ],
    )
    for sp, g in D.groupby("split"):
        write_parquet(to_table(g, "designs"), OUT / f"data/designs/designs-{sp}.parquet", False)
    write_parquet(to_table(LY, "layouts"), OUT / "data/layouts/layouts.parquet", False)
    S = S.assign(**{k: v for k, v in PROV.items()})
    S["passes_completed"] = pd.to_numeric(S.passes_completed, errors="coerce").astype("Int16")
    S["cores"] = pd.to_numeric(S.cores, errors="coerce").astype("Int16")
    for sp, g in S.groupby("split"):
        write_parquet(to_table(g, "simulations"), OUT / f"data/simulations/simulations-{sp}.parquet", False)
    role = dict(zip(S.simulation_id, S.role))
    L = L.assign(**PROV)
    L["dist_um"] = L.dist_um.astype("float32")
    L["file"] = [
        "heldout" if d in priv else (sp if role[s] == "reference" and sp in ("train", "val", "test") else "study")
        for s, d, sp in zip(L.simulation_id, L.design_id, L.split)
    ]
    counts = {}
    for f, g in L.groupby("file"):
        g = g.drop(columns="file")
        if f == "heldout":
            write_parquet(to_table(g, "labels"), HELD / "data/labels/labels-test.parquet", True)
        else:
            write_parquet(to_table(g, "labels"), OUT / f"data/labels/labels-{f}.parquet", True)
        counts[f] = len(g)
    file_of_sim = L.drop_duplicates("simulation_id").set_index("simulation_id").file
    split_of = dict(zip(D.design_id, D.split))
    # label passes (arrow): v1 rows get their split from the design, new rows carry it
    dd = pas1.column("design_id").combine_chunks().dictionary_encode()
    smap = pa.array([split_of.get(x) for x in dd.dictionary.to_pylist()])
    p1 = pas1.append_column("split", pc.take(smap, dd.indices))
    sch = SCH.arrow_schema("label_passes")
    p1 = p1.select(sch.names).cast(sch)
    pn = passN.select(sch.names).cast(sch)
    for f in sorted(set(file_of_sim)):
        sids = pa.array(file_of_sim.index[file_of_sim == f].tolist())
        t = pa.concat_tables(
            [
                x.filter(pc.is_in(x.column("simulation_id"), sids.cast(x.schema.field("simulation_id").type)))
                for x in (p1, pn)
            ]
        )
        if not t.num_rows:
            continue
        dst = (
            HELD / "data/label_passes/label_passes-test.parquet"
            if f == "heldout"
            else OUT / f"data/label_passes/label_passes-{f}.parquet"
        )
        write_parquet(t, dst, True)
    D.drop(columns=["gds_src"]).to_pickle(WORK / "designs.pkl")
    S[["simulation_id", "design_id", "status", "role", "qc_ok", "qc_convergence", "split"]].to_pickle(WORK / "sims.pkl")
    json.dump(
        dict(
            label_files=counts,
            private_designs=len(priv),
            public_test_designs=int(pub_test.sum()),
            private_non_test_by_geometry=sorted(priv - set(D.loc[D.split == "test", "design_id"])),
        ),
        open(WORK / "raw_summary.json", "w"),
        indent=1,
    )
    print("labels written", counts, flush=True)
