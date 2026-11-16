"""Overlap audit: SQuADDS_PlanarCapacitance designs vs the public SQuADDS_DB / SQuADDS_Layouts.

Matches every design of the new dataset against SQuADDS by (a) GDS file sha256 (= the Layouts manifest `artifact_id`),
(b) canonical polygon hash `geom_hash` (1 nm) of the public GDS of the candidate components, and (c) design parameters;
compares labels of matched designs (Maxwell matrix, fF); checks the Xmon (exp13) sweep against `qubit-TransmonCross-cap_matrix`
(geometry parameters AND layer stack). Public data only (pinned revisions). Writes `<work>/overlap_map.parquet` (design_id ->
squadds_db_ref JSON, read by the dataset builder) and `<work>/overlap_audit.json`.

    .venv/bin/python scripts/planar_capacitance/audit_overlap.py [--workers 4]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "build_dataset"))
from common import DL, V1, WORK, gds_info, sha256_file  # noqa: E402

DB_REPO, DB_REV = "SQuADDS/SQuADDS_DB", "0e25705f54c343fb96571ff15b6fd8375ca899aa"
LAY_REPO, LAY_REV = "SQuADDS/SQuADDS_Layouts", "78f0b913c210f1c49fbb4495db0108efb4f2b9e7"
GEN = "coupler-GeneralizedCapNInterdigital-cap_matrix"
TXC = "qubit-TransmonCross-cap_matrix"
NEW = {
    "twonet": DL / "twonet/designs",
    "synth_q3d": DL / "synth_q3d/designs",
    "campaign450": DL / "campaign_450/designs",
}
XMON_KEYS = (
    "cross_length",
    "cross_width",
    "cross_gap",
    "claw_length",
    "claw_width",
    "claw_gap",
    "ground_spacing",
    "claw_cpw_length",
    "claw_cpw_width",
)


def um(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    for u, f in (("mm", 1e3), ("um", 1.0), ("nm", 1e-3)):
        if s.endswith(u):
            try:
                return float(s[: -len(u)]) * f
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _ghash(p):
    try:
        return p, gds_info(p)["geom_hash"]
    except Exception as e:  # noqa: BLE001
        return p, f"error:{e!r}"[:200]


def stats(ours, theirs):
    a, b = np.asarray(ours, float), np.asarray(theirs, float)
    d = a - b
    rel = np.abs(d) / np.maximum(np.abs(b), 1e-12) * 100
    return dict(
        n=int(len(a)),
        median_pct=float(np.median(rel)),
        mae_fF=float(np.mean(np.abs(d))),
        rmse_fF=float(np.sqrt(np.mean(d * d))),
        max_abs_fF=float(np.max(np.abs(d))),
        n_exact=int(np.sum(np.abs(d) <= 1e-9 * np.maximum(np.abs(b), 1))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    WORK.mkdir(parents=True, exist_ok=True)
    gen = json.load(open(hf_hub_download(DB_REPO, f"{GEN}.json", repo_type="dataset", revision=DB_REV)))
    txc = json.load(open(hf_hub_download(DB_REPO, f"{TXC}.json", repo_type="dataset", revision=DB_REV)))
    lay_root = Path(
        snapshot_download(
            LAY_REPO,
            repo_type="dataset",
            revision=LAY_REV,
            allow_patterns=["metadata/*", "raw/GeneralizedCapNInterdigital/**", "raw/TransmonCross/**"],
            max_workers=a.workers,
        )
    )
    man = pd.read_parquet(lay_root / "metadata/manifest.parquet")
    print("SQuADDS:", GEN, len(gen), TXC, len(txc), "layouts manifest", len(man), flush=True)

    # our designs: v1 (layout_id, geom_hash known) + the new collections (hashed here)
    D = pd.concat(
        [
            pd.read_parquet(
                f, columns=["design_id", "collection", "layout_id", "geom_hash", "params", "split", "net_order"]
            )
            for f in glob.glob(str(V1 / "data/v1/designs/*.parquet"))
        ],
        ignore_index=True,
    )
    new = [str(g) for r in NEW.values() for g in sorted(r.glob("q3d_cap_*/cap_*.gds"))]
    with Pool(a.workers) as pool:
        nh = dict(pool.imap_unordered(_ghash, new, chunksize=32))
    rows = []
    for coll, r in NEW.items():
        for g in sorted(r.glob("q3d_cap_*/cap_*.gds")):
            rows.append(
                dict(
                    design_id=f"ref/{coll}/{g.parent.name}",
                    collection=coll,
                    layout_id="sha256:" + sha256_file(g),
                    geom_hash=nh[str(g)],
                    params=None,
                    split=None,
                    net_order=None,
                )
            )
    D = pd.concat([D, pd.DataFrame(rows)], ignore_index=True)

    # (a) file sha against the whole manifest
    art = man.set_index("artifact_id")
    by_sha = D[D.layout_id.isin(art.index)]
    # (b) geom_hash of the public GDS of the candidate components
    cand = man[man.component_name.isin(["GeneralizedCapNInterdigital", "TransmonCross"])]
    paths = [str(lay_root / p) for p in cand.gds_path if (lay_root / p).exists()]
    with Pool(a.workers) as pool:
        gh = dict(pool.imap_unordered(_ghash, paths, chunksize=64))
    cand = cand.assign(geom_hash=[gh.get(str(lay_root / p)) for p in cand.gds_path])
    n_missing_gds = int(cand.geom_hash.isna().sum())
    cg = cand.dropna(subset=["geom_hash"]).drop_duplicates("geom_hash").set_index("geom_hash")
    by_geom = D[D.geom_hash.isin(cg.index)]

    # SQuADDS_DB row index per source_id (generalized config) and per Layouts design_id (TransmonCross)
    gen_ix = {r["notes"].get("source_id"): i for i, r in enumerate(gen)}
    refs = {}
    for _, r in pd.concat([by_sha, by_geom]).drop_duplicates("design_id").iterrows():
        m = art.loc[r.layout_id] if r.layout_id in art.index else cg.loc[r.geom_hash]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        ref = dict(
            repo=DB_REPO,
            revision=DB_REV,
            config=m.simulation_config,
            source_id=m.source_id,
            row_index=gen_ix.get(m.source_id) if m.simulation_config == GEN else None,
            layouts_repo=LAY_REPO,
            layouts_revision=LAY_REV,
            layouts_design_id=m.design_id,
            gds_path=m.gds_path,
            match="file_sha256" if r.layout_id in art.index else "geom_hash",
        )
        refs[r.design_id] = ref
    M = pd.DataFrame(
        [
            dict(design_id=d, collection=d.split("/")[1], squadds_db_ref=json.dumps(v, sort_keys=True), **v)
            for d, v in refs.items()
        ]
    )
    out = dict(
        revisions={DB_REPO: DB_REV, LAY_REPO: LAY_REV},
        squadds_rows={GEN: len(gen), TXC: len(txc)},
        layouts_manifest_rows=len(man),
        candidate_public_gds_hashed=len(paths),
        candidate_gds_not_downloaded=n_missing_gds,
    )
    split_of = dict(zip(D.design_id, D.split))
    if len(M):
        M["split_v1"] = M.design_id.map(split_of)
        out["overlap_by_collection"] = {
            c: dict(
                designs=int((D.collection == c).sum()),
                matched=int(len(g)),
                by_file_sha=int((g.match == "file_sha256").sum()),
                by_geom_hash_only=int((g.match == "geom_hash").sum()),
                configs=g.config.value_counts().to_dict(),
                split_v1=g.split_v1.value_counts(dropna=False).to_dict(),
            )
            for c, g in M.groupby("collection")
        }
        # one-to-one?
        out["matched_squadds_rows"] = int(M.source_id.nunique())
        out["our_designs_per_squadds_row_max"] = int(M.groupby("source_id").size().max())
    out["squadds_campaign_rows"] = pd.Series([r["notes"].get("source_campaign") for r in gen]).value_counts().to_dict()
    out["unmatched_squadds_generalized_rows"] = int(len(gen) - (M.source_id.isin(gen_ix).sum() if len(M) else 0))
    out["param_check"] = _param_check(D, M, gen, gen_ix)
    out["labels"] = _labels(M, gen, gen_ix, split_of)
    out["transmoncross_vs_xmon"] = _xmon(D, txc, cand)
    M.to_parquet(WORK / "overlap_map.parquet", index=False)
    json.dump(out, open(WORK / "overlap_audit.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("param_check",)}, indent=1, default=str)[:6000])


def _param_check(D, M, gen, gen_ix):
    """For file-matched interdigital designs: do cap_options equal the SQuADDS design_options (numeric, um)?"""
    if not len(M):
        return {}
    par = dict(zip(D.design_id, D.params))
    n = eq = 0
    diff_keys = {}
    for d, sid in zip(M.design_id, M.source_id):
        if sid not in gen_ix or not par.get(d):
            continue
        ours = flat(json.loads(par[d]).get("cap_options", {}))
        theirs = flat(gen[gen_ix[sid]]["design"]["design_options"])
        keys = sorted(set(ours) & set(theirs))
        bad = [
            k
            for k in keys
            if (um(ours[k]) is not None and um(theirs[k]) is not None and abs(um(ours[k]) - um(theirs[k])) > 1e-9)
            or (um(ours[k]) is None and str(ours[k]) != str(theirs[k]))
        ]
        n += 1
        eq += not bad
        for k in bad:
            diff_keys[k] = diff_keys.get(k, 0) + 1
        for k in set(ours) ^ set(theirs):
            diff_keys[f"only_in_{'ours' if k in ours else 'squadds'}:{k}"] = (
                diff_keys.get(f"only_in_{'ours' if k in ours else 'squadds'}:{k}", 0) + 1
            )
    return dict(
        compared=n, all_shared_params_equal=eq, differing_keys=dict(sorted(diff_keys.items(), key=lambda x: -x[1])[:20])
    )


def _labels(M, gen, gen_ix, split_of):
    """Our final Maxwell matrix vs the SQuADDS maxwell_matrix of the same design (non-test: statistics; test: counts only)."""
    if not len(M):
        return {}
    want = set(M.design_id)
    files = glob.glob(str(V1 / "data/v1/labels/*.parquet")) + [
        str(V1 / "private_heldout/data/v1/labels/labels-test.parquet")
    ]
    L = pd.concat(
        [
            pd.read_parquet(
                f, columns=["design_id", "net_a", "net_b", "C_fF"], filters=[("design_id", "in", list(want))]
            )
            for f in files
        ],
        ignore_index=True,
    )
    ent = {(d, frozenset((a, b))): c for d, a, b, c in zip(L.design_id, L.net_a, L.net_b, L.C_fF)}
    res = {"mutual": ([], []), "self_signal": ([], []), "self_ground": ([], [])}
    per_split = {}
    test_exact, test_n = 0, 0
    for d, sid, _coll in zip(M.design_id, M.source_id, M.collection):
        if sid not in gen_ix:
            continue
        mm = gen[gen_ix[sid]]["sim_results"].get("maxwell_matrix") or {}
        pairs = [("N", "S"), ("N", "G"), ("S", "G"), ("N", "N"), ("S", "S"), ("G", "G")]
        vals = [(a, b, ent.get((d, frozenset((a, b)))), mm.get(f"C_{a}_{b}")) for a, b in pairs]
        vals = [v for v in vals if v[2] is not None and v[3] is not None]
        if not vals:
            continue
        sp = split_of.get(d)
        per_split[sp] = per_split.get(sp, 0) + 1
        if sp == "test":
            test_n += 1
            test_exact += all(abs(o - t) <= 1e-9 * max(abs(t), 1) for _, _, o, t in vals)
            continue
        differs = any(abs(o - t) > 1e-9 * max(abs(t), 1) for _, _, o, t in vals)
        for a, b, o, t in vals:
            k = "mutual" if a != b else ("self_ground" if a == "G" else "self_signal")
            res[k][0].append(o)
            res[k][1].append(t)
            if differs:
                res.setdefault(f"{k}_differing_designs_only", ([], []))
                res[f"{k}_differing_designs_only"][0].append(o)
                res[f"{k}_differing_designs_only"][1].append(t)
        per_split["non_test_designs_with_any_differing_entry"] = (
            per_split.get("non_test_designs_with_any_differing_entry", 0) + differs
        )
    out = {k: stats(*v) for k, v in res.items() if v[0]}
    out["designs_compared_by_split_v1"] = per_split
    out["test_designs_matched"] = test_n
    out["test_designs_bit_identical_matrix"] = test_exact
    return out


def _xmon(D, txc, cand):
    """Xmon sweep (exp13) vs SQuADDS TransmonCross: nearest geometry in parameter space, and the layer stack."""
    ours = []
    for _d, p in zip(D.design_id[D.collection == "exp13"], D.params[D.collection == "exp13"]):
        o = json.loads(p).get("cap_options", {})
        cp = o.get("connection_pads", {}).get("readout", {})
        v = [um(o.get(k)) for k in XMON_KEYS[:3]] + [um(cp.get(k)) for k in XMON_KEYS[3:]]
        if all(x is not None for x in v):
            ours.append(v)
    theirs = []
    stacks, sims = {}, {}
    for r in txc:
        o = r["design"]["design_options"]
        cp = o.get("connection_pads", {}).get("readout", {})
        v = [
            um(o.get("cross_length")),
            um(o.get("cross_width")),
            um(o.get("cross_gap")),
            um(cp.get("claw_length")),
            um(cp.get("claw_width")),
            um(cp.get("claw_gap")),
            um(cp.get("ground_spacing")),
            um(cp.get("claw_cpw_length")),
            um(cp.get("claw_cpw_width")),
        ]
        if all(x is not None for x in v):
            theirs.append(v)
        s = json.dumps(r["sim_options"].get("layer_stack"))
        stacks[s] = stacks.get(s, 0) + 1
        sims[r["sim_options"].get("simulator")] = sims.get(r["sim_options"].get("simulator"), 0) + 1
    A, B = np.array(ours), np.array(theirs)
    out = dict(
        exp13_designs_with_params=len(A),
        transmoncross_rows_with_params=len(B),
        squadds_layer_stack=stacks,
        squadds_simulator_field=sims,
        ours_stack="Si eps_r 11.45, 350 um substrate, zero-thickness sheets, Q3D 13 GHz",
    )
    if len(A) and len(B):
        rel = np.abs(A[:, None, :] - B[None, :, :]) / np.maximum(np.abs(B[None, :, :]), 1e-9)
        best = rel.max(-1).min(-1)
        out.update(
            exact_param_matches=int(np.sum(best < 1e-6)),
            within_1pct=int(np.sum(best < 0.01)),
            nearest_max_rel_diff_median=float(np.median(best)),
            nearest_max_rel_diff_min=float(best.min()),
            ranges_ours={k: [float(A[:, i].min()), float(A[:, i].max())] for i, k in enumerate(XMON_KEYS)},
            ranges_squadds={k: [float(B[:, i].min()), float(B[:, i].max())] for i, k in enumerate(XMON_KEYS)},
        )
    xg = set(D.geom_hash[D.collection == "exp13"].dropna())
    out["geom_hash_matches"] = int(cand[(cand.component_name == "TransmonCross")].geom_hash.isin(xg).sum())
    out["merge_decision"] = (
        "separate config: geometry and layer stack must both match; SQuADDS TransmonCross rows record no "
        "layer_stack (qiskit-metal default substrate 750 um, SQuADDS preset squadds_hfss_v1) vs 350 um here"
    )
    return out


if __name__ == "__main__":
    main()
