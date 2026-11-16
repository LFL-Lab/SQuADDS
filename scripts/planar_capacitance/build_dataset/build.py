"""Build SQuADDS/SQuADDS_PlanarCapacitance locally (nothing is uploaded). See README.md in this folder.

.venv/bin/python scripts/planar_capacitance/audit_overlap.py --workers 4      # writes the overlap map first
.venv/bin/python scripts/planar_capacitance/build_dataset/build.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import book
import checks
import numpy as np
import pandas as pd
import processed
import raw
from common import HELD, OUT, WORK, code_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", default="raw,processed,book,checks")
    ap.add_argument("--skip-exp612-check", action="store_true")
    a = ap.parse_args()
    steps = a.steps.split(",")
    t0 = time.time()
    timing = {}
    if "raw" in steps:
        for d in (OUT / "data", HELD / "data"):
            shutil.rmtree(d, ignore_errors=True)
        D, LY, S, L, pas1, passN = raw.build(a.workers)
        raw.write(D, LY, S, L, pas1, passN)
        del pas1, passN, L
        timing["raw_s"] = time.time() - t0
    D = pd.read_pickle(WORK / "designs.pkl")
    split_of = dict(zip(D.design_id, D.split))
    if "processed" in steps:
        t = time.time()
        processed.build(split_of)
        timing["processed_s"] = time.time() - t
    if "book" in steps:
        t = time.time()
        book.write_book(D)
        book.heldout_card()
        timing["book_s"] = time.time() - t
    res = json.load(open(WORK / "checks.json")) if (WORK / "checks.json").exists() and "checks" not in steps else {}
    if "checks" in steps:
        t = time.time()
        res = dict(
            counts=checks.check_counts(),
            no_test_labels=checks.check_no_test_labels(),
            superset=checks.check_superset(),
            processed_bit_equal=checks.check_processed(),
            labels_vs_sources=checks.check_labels_vs_sources(),
            qc=checks.qc_summary(),
        )
        if not a.skip_exp612_check:
            res["exp612_vs_tlsq_reader"] = checks.check_exp612(a.workers)
        res["sizes"] = checks.sizes()
        timing["checks_s"] = time.time() - t
        json.dump(res, open(WORK / "checks.json", "w"), indent=1, default=_np)
    if "book" in steps or "checks" in steps:
        cnt = res.get("counts", {}).get("rows", [])
        md = "| Table I row | train / val / test (Table I) | dataset |\n|---|---|---|\n" + "\n".join(
            f"| {r['table_i_row']} | {' / '.join(map(str, r['table_i'] or ()))} | {' / '.join(map(str, r['dataset']))} |"
            for r in cnt
        )
        n_ref = int(D.squadds_db_ref.notna().sum())
        sims = pd.read_pickle(WORK / "sims.pkl")
        book.write_card(dict(n_designs=len(D), n_simulations=len(sims), n_squadds_ref=n_ref, counts_md=md))
        timing["total_s"] = time.time() - t0
        extra = dict(
            build_code_sha256=code_sha(),
            timing_s=timing,
            checks=res,
            raw=json.load(open(WORK / "raw_summary.json")),
            processed=json.load(open(WORK / "processed_summary.json"))
            if (WORK / "processed_summary.json").exists()
            else None,
        )
        book.write_manifest("SQuADDS/SQuADDS_PlanarCapacitance", OUT, extra)
        book.write_manifest("SQuADDS/SQuADDS_PlanarCapacitance-heldout", HELD, dict(build_code_sha256=code_sha()))
    print(json.dumps(timing, indent=1), flush=True)


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    main()
