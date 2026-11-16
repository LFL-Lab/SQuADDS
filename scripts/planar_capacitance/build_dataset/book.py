"""Bookkeeping tables, dataset card, changelog and manifest."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pyarrow as pa
from common import DL, DSV, HELD, OUT, PUBLIC_TEST_COLLECTIONS, SCH, WORK, sha256_file, write_parquet

SAIKAT = ["exp6", "exp7", "exp8", "exp9", "exp10", "exp11", "exp12", "margin_paddle"]
TEST_RULES = {
    "infamily_test": "all mutual entries",
    "newfam926": "test only for models trained without exp13-15",
    "grid_qubit_1": "within 100 um and beyond 1 mm separately",
    "grid_qubit_5": "within 100 um and beyond 1 mm separately",
    "grid_qubit_9": "within 100 um and beyond 1 mm separately",
}


def write_book(D: pd.DataFrame):
    contributors = pd.DataFrame(
        [
            dict(
                contributor_id="saikat_das",
                name="Saikat Das",
                affiliation="Levenson-Falk Lab, USC",
                role="generator, simulator",
                collections=SAIKAT,
                notes="exp6-12 designs and Q3D solves, exp12 full re-export; paddle margin designs and solves",
            ),
            dict(
                contributor_id="sadman_shanto",
                name="Sadman Ahmed Shanto",
                affiliation="Levenson-Falk Lab, USC",
                role="generator, simulator, curator",
                collections=sorted(set(D.collection) - set(SAIKAT)),
                notes="all other designs and Q3D solves, the lattice layouts (grid), priors, embeddings, curation",
            ),
        ]
    )
    write_parquet(pa.Table.from_pandas(contributors, preserve_index=False), OUT / "data/contributors.parquet", False)
    write_parquet(
        pa.Table.from_pandas(
            pd.DataFrame(
                [
                    dict(
                        licence_id=SCH.LICENSE,
                        spdx=SCH.LICENSE,
                        url="https://creativecommons.org/licenses/by/4.0/",
                        status="confirmed by the author 2026-10-29",
                    )
                ]
            ),
            preserve_index=False,
        ),
        OUT / "data/licences.parquet",
        False,
    )
    ct = D.groupby(["collection", "split"]).size().unstack(fill_value=0)
    rules = [
        (
            "exp6-exp12",
            "sha256(group)[:8] mod 100 -> <10 test, <20 val, else train; group = family + rounded geometry signature "
            "(exp6-8 swept-parameter tuple); i.i.d. within the sweeps",
        ),
        ("exp13-exp15", "926 test fixed before labels; the 828-design pool split 90/10 by sha256(exp/design) mod 100"),
        ("random_v1, synth_q3d", "sha256(parent key) mod 100 -> 0-79 train, 80-89 val, 90-99 sealed test"),
        ("twonet", "declared before labels: train 750 / val 100 / test 150 (held-out parameter region)"),
        ("campaign450", "B1, B2 train; B3 validation; B4 and B5 (validation-only rings) excluded"),
        ("external sets", "atlas, qm116, p3rand36, wonky_v1, largen, lattices: test only"),
    ]
    sp = pd.DataFrame(
        [
            dict(
                split_version="paper_v4",
                scope=s,
                rule=r,
                frozen_date="2026-10-16",
                notes="train/val = the paper's P2d training and validation shards; test = the X1 manifest (5,902 layouts); "
                "split 'excluded' = in a Table I collection but not used (label QC, convergence, bucket); "
                "split_v1 keeps the v1 release value",
            )
            for s, r in rules
        ]
    )
    sp["counts_json"] = json.dumps({c: {k: int(v) for k, v in r.items() if v} for c, r in ct.iterrows()})
    write_parquet(pa.Table.from_pandas(sp, preserve_index=False), OUT / "data/splits.parquet", False)
    man = json.load(open(DL / "v4test/x1_manifest.json"))
    from common import key_to_design_id

    by = {}
    for e in man:
        by.setdefault(e["cohort"], []).append(key_to_design_id(e["key"]))
    pub = set(D.loc[(D.split == "test") & D.collection.isin(PUBLIC_TEST_COLLECTIONS), "design_id"])

    def vis(ids):
        n = sum(i in pub for i in ids)
        if not n:
            return "private (SQuADDS_PlanarCapacitance-heldout)"
        if n == len(ids):
            return "public (already public in SQuADDS/SQuADDS_DB)"
        return (
            f"mixed: {n} exp6-8 interdigital designs public (already public in SQuADDS/SQuADDS_DB), "
            f"{len(ids) - n} private (SQuADDS_PlanarCapacitance-heldout)"
        )

    ts = pd.DataFrame(
        [
            dict(
                test_set_id=c,
                design_ids=ids,
                n=len(ids),
                entries_rule=TEST_RULES.get(c, "all mutual entries"),
                label_visibility=vis(ids),
            )
            for c, ids in sorted(by.items())
        ]
    )
    write_parquet(pa.Table.from_pandas(ts, preserve_index=False), OUT / "data/test_sets.parquet", False)
    reg = pd.DataFrame(
        [
            dict(
                id=SCH.EMBEDDING_VERSION,
                kind="embedding",
                table="geometric_embedding",
                title="conductor (60 + missing flag), pair (42) and global (4) geometric vectors of the planar-capacitance GNN",
                code="tl-squadds geomfast stage (scene, localtok, blocks); SQuADDS squadds.layouts.capacitance_graph",
            ),
            dict(
                id=SCH.COARSE_PRIOR_VERSION,
                kind="prior",
                table="physics_prior",
                title="thin-film method-of-moments prior, single solve at 0.375 panels per gap (v1 frozen solver, alpha 0.49685)",
                code="tl-squadds src/tlsq/embed/fastprior_loc.py micro_mom_loc (frozen geomfast vendor)",
            ),
            dict(
                id=SCH.OPTIMIZED_PRIOR_VERSION,
                kind="prior",
                table="optimized_physics_prior",
                title="C^R = exp(2 log c(3) - log c(1.5)) on the corrected stack: v2m corrections + substrate far field (arm b)",
                code="priorv2.py installed(policy=fc, dip=True, margin=True) + farfield.py install(b); fastprior_pkg exact_compiled",
            ),
        ]
    )
    write_parquet(pa.Table.from_pandas(reg, preserve_index=False), OUT / "data/registry.parquet", False)


def configs_yaml() -> str:
    lines = ["configs:"]
    groups = {
        "designs": "data/designs/designs-*.parquet",
        "layouts": "data/layouts/layouts.parquet",
        "simulations": "data/simulations/simulations-*.parquet",
        "labels": "data/labels/labels-*.parquet",
        "label_passes": "data/label_passes/label_passes-*.parquet",
        "geometric_embedding_nodes": "data/geometric_embedding/nodes-*.parquet",
        "geometric_embedding_edges": "data/geometric_embedding/edges-*.parquet",
        "geometric_embedding_globals": "data/geometric_embedding/globals-*.parquet",
        "physics_prior": "data/physics_prior/physics_prior-*.parquet",
        "optimized_physics_prior": "data/optimized_physics_prior/optimized_physics_prior-*.parquet",
        "model_input": "data/model_input/model_input-*.parquet",
    }
    for name, pat in groups.items():
        files = sorted(OUT.glob(pat))
        lines += [f"- config_name: {name}", "  data_files:"]
        for f in files:
            sp = f.stem.split("-")[-1] if "-" in f.stem else "train"
            lines += [f"  - split: {sp if name != 'layouts' else 'all'}", f"    path: {f.relative_to(OUT)}"]
    for name in ("splits", "test_sets", "contributors", "licences", "registry"):
        lines += [f"- config_name: {name}", "  data_files:", "  - split: all", f"    path: data/{name}.parquet"]
    return "\n".join(lines)


def write_card(summary: dict):
    cnt = summary.get("counts_md", "")
    card = (
        f"""---
license: cc-by-4.0
pretty_name: SQuADDS Planar Capacitance
task_categories:
- tabular-regression
- graph-ml
tags:
- superconducting-qubits
- capacitance
- gds
- q3d
size_categories:
- 10K<n<100K
{configs_yaml()}
---

# SQuADDS_PlanarCapacitance

Layout (GDS) to full Maxwell capacitance matrix for superconducting-circuit components: {summary.get("n_designs", "?"):,} designs,
{summary.get("n_simulations", "?"):,} solver runs (Ansys Q3D), the geometric embedding `{SCH.EMBEDDING_VERSION}`, a coarse and an
optimized thin-film physics prior, and the input the planar-capacitance models consume. Licence CC-BY-4.0.

## Conventions
- Capacitance in **fF**, Maxwell signs (diagonal > 0, off-diagonal < 0; `mutual_fF = -C_fF`), long format, `a_idx <= b_idx` in the
  design's `net_order`, **ground `G` first**. Nets are never merged: N is inferred from the layout (N counts ground).
- Label = last complete adaptive pass of the Q3D export (exp10 unfed S/G swap corrected; exp12 from the full re-export).
- Stack of every reference label: silicon eps_r 11.45, 350 um, zero-thickness sheets, 13 GHz adaptive solve.
- `geom_hash` = sha256 of the canonical polygon multiset (1 nm) = the uniqueness key across SQuADDS. Designs already in
  `SQuADDS/SQuADDS_DB` carry `squadds_db_ref` (repo, revision, config, source_id, row index); the SQuADDS view joins them
  instead of storing them twice.
- Every processed row carries its `prior_version` / `embedding_version`, the code sha256 and its source.
- **Test labels** (split `test`, 5,902 layouts of the paper's Table I) are not in this repository, with one exception: their
  layouts, embeddings and priors are; their labels are in the private `SQuADDS_PlanarCapacitance-heldout`. The exception is the
  1,904 interdigital in-family test designs (exp6 164, exp7 128, exp8 1,612): their matrices are already public in
  `SQuADDS/SQuADDS_DB`, so their labels and passes are here (`labels-test`, `label_passes-test`, split kept `test`, design
  `notes` = "labels public: already public in SQuADDS/SQuADDS_DB"). Scores on them are not blind.

## Tables
"""
        + "\n".join(f"- `{k}`: {v}" for k, v in SCH.DESCRIPTIONS.items())
        + f"""
- bookkeeping: `splits`, `test_sets`, `contributors`, `licences`, `registry`.

## Counts (Table I of the paper)
{cnt}

## Contributors
Saikat Das (exp6-12 designs and Q3D, paddle margin); Sadman Ahmed Shanto (all other designs incl. the lattice layouts, Q3D,
priors, embeddings, curation).

## Citation
TODO (paper in preparation).
"""
    )
    (OUT / "README.md").write_text(card)
    (OUT / "CHANGELOG.md").write_text(f"""# Changelog

## {DSV} ({date.today().isoformat()}, local build, not uploaded)
- New repository `SQuADDS/SQuADDS_PlanarCapacitance`.
- Added collections: two-net (1,000 designs), mixed-shape synthetic Q3D (1,000), campaign 450 (490); their Q3D runs.
- Splits follow the paper (P2d shards train/val, X1 manifest test); `split_v1` keeps the v1 value. Labels of designs whose
  split became `test` (e.g. random_v1 sealed 110, grid_qubit_1) moved to the heldout repository; the exp6-8 in-family test
  labels stay public (already public in `SQuADDS/SQuADDS_DB`).
- New processed tables: `geometric_embedding` ({SCH.EMBEDDING_VERSION}), `physics_prior` ({SCH.COARSE_PRIOR_VERSION}),
  `optimized_physics_prior` ({SCH.OPTIMIZED_PRIOR_VERSION}), `model_input`.
- Label QC flags (symmetry, signs, closure, PD, convergence) recomputed for every simulation; nothing dropped.
- `squadds_db_ref` links the {summary.get("n_squadds_ref", "?"):,} designs already in `SQuADDS/SQuADDS_DB`.
""")


def write_manifest(name: str, root, extra: dict):
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    man = dict(
        dataset=name,
        dataset_version=DSV,
        schema_version=SCH.SCHEMA_VERSION,
        n_files=len(files),
        total_bytes=sum(f["bytes"] for f in files),
        **extra,
        files=files,
    )
    json.dump(man, open(root / "MANIFEST.json", "w"), indent=1, default=str)
    return man


def heldout_card():
    (HELD / "README.md").write_text(f"""---
license: cc-by-4.0
private: true
---
# SQuADDS_PlanarCapacitance-heldout (private)

Test labels of `SQuADDS/SQuADDS_PlanarCapacitance` {DSV}: `data/labels/labels-test.parquet` and
`data/label_passes/label_passes-test.parquet`, same schema as the public `labels` / `label_passes`. Evaluation data only: never
use them to fit, tune or select. Also holds the labels of non-test designs whose geometry equals a test design, if any. The
exp6-8 interdigital in-family test labels (1,904 designs) are not here: they are already public in `SQuADDS/SQuADDS_DB` and ship
in the public repository.
""")
    WORK.mkdir(parents=True, exist_ok=True)
