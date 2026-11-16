"""Rebuild tl-squadds dataset layouts with the SQuADDS components and compare them with the dataset GDS.

For a stratified sample of every family the design is rebuilt from its stored options, exported with the dataset's
GDS convention (1 nm unit; ground with its etched pocket on (1, 0); one polygon per signal body on (1, 10); 2 um port
stubs on (2, 0) / (3, 0); chip = component extent plus m ground gaps) and compared per layer with the dataset file:
polygon count, symmetric-difference area (um^2) and a canonical 1 nm polygon hash, plus the conductor count.

The export is ``squadds.layouts.dataset_export`` (both conventions: squadds2-helpers for exp6-12, the families
generator for exp13-15); this script rebuilds each record's component and passes the record's margin factor and, for
exp13-15, the generator's unrounded ground gap.

Usage:
    QISKIT_METAL_HEADLESS=1 python scripts/planar_capacitance/verify_component_gds.py \
        --data-root /path/to/tl_squadds/data_local --per-family 60 --workers 4 --out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("QISKIT_METAL_HEADLESS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

FAMILIES = {
    "interdigital": ("designs_ood/data/design_files", ("exp6", "exp7", "exp8")),
    "paddle": ("designs_ood/data/design_files", ("exp9",)),
    "concentric": ("designs_ood/data/design_files", ("exp10",)),
    "star": ("designs_ood/data/design_files", ("exp11",)),
    "multipad_star": ("designs_ood/data/design_files", ("exp12",)),
    "xmon": ("newfam/design_files", ("exp13",)),
    "pocket": ("newfam/design_files", ("exp14",)),
    "shielded_concentric": ("newfam/design_files", ("exp15",)),
}
# exp6 and exp7 were generated with the first squadds2-helpers interdigital rule (50664c7): tangent spine ledges only
TANGENT_LEDGE_EXPS = {"exp6", "exp7"}


# ----------------------------------------------------------------------------------------------- sampling
def _um(value) -> float:
    text = str(value).strip()
    if text.endswith("um"):
        return float(text[:-2])
    if text.endswith("mm"):
        return 1e3 * float(text[:-2])
    return float(text)


def _sign(a, b) -> str:
    a, b = _um(a), _um(b)
    return "=" if abs(a - b) < 1e-9 else (">" if a > b else "<")


def stratum(family: str, cfg: dict) -> str:
    o = cfg["cap_options"]
    if family == "interdigital":
        return f"{cfg['exp']}|fc={o['finger_count']}"
    if family == "paddle":
        pairs = ("finger_width", "spine_width", "cpw_width", "cpw_gap", "cpw_length", "gap_ground", "cpw_xpos_offset")
        return "".join(
            _sign(o.get(f"north_{k}", o.get("finger_width")), o.get(f"south_{k}", o.get("finger_width"))) for k in pairs
        )
    if family == "concentric":
        shape = "".join(
            "c" if o[f"{k}_radius_x"] == o[f"{k}_radius_y"] else "e"
            for k in ("inner", "outer_inner", "outer_outer", "ground_inner")
        )
        return f"feed={bool(o['inner_feed'])}|{shape}"
    if family in ("star", "multipad_star"):
        pads = sum(int(x) for x in str(o["pad_enables"]).split(","))
        return f"pads={pads}|{o['valley_ground']}"
    params = cfg["generator"]["params"]
    if family == "xmon":
        return f"claws={2 if params['second_claw'] else 1}" + (
            f"|loc={params['second_claw_location']}" if params["second_claw"] else ""
        )
    if family == "pocket":
        return f"pads={2 if params['second_cp'] else 1}"
    if family == "shielded_concentric":
        return f"guards={len(params['guards'])}|" + "".join("g" if g["grounded"] else "f" for g in params["guards"])
    raise ValueError(family)


def collect(data_root: Path, family: str) -> list[tuple[str, Path]]:
    sub, exps = FAMILIES[family]
    out = []
    for exp in exps:
        for folder in sorted((data_root / sub / exp).glob("q3d_cap_*")):
            tid = folder.name.split("_")[-1]
            if (folder / f"cap_{tid}.gds").is_file() and (folder / f"cap_{tid}.json").is_file():
                out.append((exp, folder))
    return out


def stratified_sample(family: str, items: list, n: int, rng: random.Random) -> list:
    """At least one design per stratum, the rest spread round-robin over strata (seeded)."""
    by = defaultdict(list)
    for exp, folder in items:
        cfg = json.loads((folder / f"cap_{folder.name.split('_')[-1]}.json").read_text())
        cfg["exp"] = exp
        by[stratum(family, cfg)].append((exp, folder))
    for v in by.values():
        rng.shuffle(v)
    picked, keys = [], sorted(by)
    while len(picked) < min(n, len(items)):
        for k in keys:
            if by[k] and len(picked) < max(n, len(keys)):
                picked.append(by[k].pop())
        if not any(by.values()):
            break
    return picked


# ----------------------------------------------------------------------------------------------- rebuild
def dataset_component(family: str, cfg: dict):
    """The SQuADDS component of a dataset record (stored options) in a scratch design, and its export overrides."""
    from qiskit_metal.designs.design_planar import DesignPlanar

    from squadds.components import get_component_class
    from squadds.components.shielded_cap_concentric import shielded_options_from_generator
    from squadds.layouts.dataset_export import CLASS_OF_FAMILY, generator_ground_gap_um

    mx, my = float(cfg["margin_factor_ew"]), float(cfg["margin_factor_ns"])
    if mx != my:
        raise ValueError(f"unequal margin factors {mx} / {my}")
    cls = get_component_class(CLASS_OF_FAMILY[family]) if family != "pocket" else None
    design = DesignPlanar({}, overwrite_enabled=True)
    gap = None
    if family in ("xmon", "pocket", "shielded_concentric"):
        # the generator set the chip margin from its unrounded parameters
        p = cfg["generator"]["params"]
        gap = generator_ground_gap_um(family, p)
        if family == "pocket":
            from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

            cls = TransmonPocket
        opts = shielded_options_from_generator(p) if family == "shielded_concentric" else cfg["cap_options"]
        name = {"xmon": "xmon", "pocket": "pocket", "shielded_concentric": "shield"}[family]
    else:
        opts, name = dict(cfg["cap_options"]), cfg["cap_instance_name"]
        if cfg.get("exp") in TANGENT_LEDGE_EXPS:
            opts["spine_ledge_rule"] = "tangent"
    comp = cls(design, name, options=opts)
    return comp, {"margin_gaps": mx, "ground_gap_um": gap}


def export_dataset(family: str, cfg: dict, star_style: str | None = None):
    from squadds.layouts.dataset_export import dataset_export_cell

    comp, kw = dataset_component(family, cfg)
    return dataset_export_cell(comp, family=family, export_style=star_style, **kw)


# ----------------------------------------------------------------------------------------------- comparison
def _canonical(points) -> tuple:
    import numpy as np

    pts = np.round(np.asarray(points)).astype(np.int64)
    keep = np.any(pts != np.roll(pts, 1, axis=0), axis=1)
    pts = pts[keep]
    changed = True
    while changed and len(pts) > 3:
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        cross = (pts[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1]) - (pts[:, 1] - prev[:, 1]) * (
            nxt[:, 0] - prev[:, 0]
        )
        changed = bool(np.any(cross == 0))
        pts = pts[cross != 0]
    area2 = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area2 < 0:
        pts = pts[::-1]
    start = min(range(len(pts)), key=lambda i: (pts[i, 0], pts[i, 1]))
    return tuple(map(tuple, np.roll(pts, -start, axis=0).tolist()))


def layer_hash(polys) -> str:
    h = hashlib.sha1()
    for c in sorted(_canonical(p.points) for p in polys):
        h.update(repr(c).encode())
    return h.hexdigest()[:16]


def raw_key(polys) -> list:
    """Polygons as written (1 nm vertices in file order, start vertex and closing point included), sorted: equal raw keys
    give equal ``squadds.ml.capacitance.qc.geom_hash``."""
    import numpy as np

    return sorted(np.round(p.points).astype(np.int64).tobytes() for p in polys)


def compare(ref_cell, new_cell) -> dict:
    import gdstk

    def by_layer(cell):
        out = defaultdict(list)
        for p in cell.polygons:
            out[(p.layer, p.datatype)].append(p)
        return out

    ref, new = by_layer(ref_cell), by_layer(new_cell)
    layers = {}
    for key in sorted(set(ref) | set(new)):
        a, b = ref.get(key, []), new.get(key, [])
        xor = gdstk.boolean(a, b, "xor") if a and b else (a or b)
        area = sum(p.area() for p in xor) / 1e6
        layers[f"{key[0]}/{key[1]}"] = {
            "n_ref": len(a),
            "n_new": len(b),
            "raw_equal": raw_key(a) == raw_key(b),
            "xor_um2": area,
            "hash_ref": layer_hash(a),
            "hash_new": layer_hash(b),
            "hash_equal": layer_hash(a) == layer_hash(b),
        }
    n_ref = len(ref.get((1, 10), [])) + (1 if ref.get((1, 0)) else 0)
    n_new = len(new.get((1, 10), [])) + (1 if new.get((1, 0)) else 0)
    chip = [
        [round(float(v) / 1e3, 3) for v in c.get((1, 0), [None])[0].bounding_box()[1]] if c.get((1, 0)) else None
        for c in (ref, new)
    ]
    return {
        "layers": layers,
        "conductors_ref": n_ref,
        "conductors_new": n_new,
        "chip_ref_um": chip[0],
        "chip_new_um": chip[1],
    }


def _score(ref, new) -> dict:
    rec = compare(ref, new)
    layers = rec["layers"]
    rec["max_xor_um2"] = max(v["xor_um2"] for v in layers.values())
    rec["count_ok"] = all(v["n_ref"] == v["n_new"] for v in layers.values())
    rec["exact"] = rec["count_ok"] and all(v["hash_equal"] for v in layers.values())
    rec["raw_equal"] = rec["count_ok"] and all(v["raw_equal"] for v in layers.values())
    rec["exact_signal"] = all(v["hash_equal"] for k, v in layers.items() if k.startswith("1/10"))
    rec["conductors_ok"] = rec["conductors_ref"] == rec["conductors_new"]
    return rec


def star_styles(exp: str, star_style: str) -> tuple[str, ...]:
    """Export styles tried for a star design (auto): exp12 = gdspy re-export; exp11 = original, else re-export."""
    if star_style != "auto":
        return (star_style,)
    return ("reexport_gdspy",) if exp == "exp12" else ("original", "reexport")


def check_one(args) -> dict:
    family, exp, folder, tmp = args[:4]
    star_style = args[4] if len(args) > 4 else "auto"
    import gdstk

    tid = Path(folder).name.split("_")[-1]
    cfg = json.loads((Path(folder) / f"cap_{tid}.json").read_text())
    cfg["exp"] = exp
    rec = {"family": family, "exp": exp, "id": tid, "stratum": stratum(family, cfg)}
    t0 = time.time()
    try:
        ref = gdstk.read_gds(str(Path(folder) / f"cap_{tid}.gds"), 1e-9).top_level()[0]
        if family in ("star", "multipad_star"):
            best = None
            for style in star_styles(exp, star_style):
                r = _score(ref, export_dataset(family, cfg, style)) | {"export_style": style}
                if best is None or (r["exact"], -r["max_xor_um2"]) > (best["exact"], -best["max_xor_um2"]):
                    best = r
                if r["exact"]:
                    break
            rec.update(best)
        else:
            rec.update(_score(ref, export_dataset(family, cfg)))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=4)
    rec["seconds"] = time.time() - t0
    return rec


def summarize(records: list[dict], tol_um2: float) -> dict:
    out = {}
    for fam in FAMILIES:
        rs = [r for r in records if r["family"] == fam]
        if not rs:
            continue
        ok = [r for r in rs if "error" not in r]
        out[fam] = {
            "n": len(rs),
            "errors": len(rs) - len(ok),
            "strata": dict(Counter(r["stratum"] for r in rs)),
            "exact_hash_all_layers": sum(r["exact"] for r in ok),
            "exact_hash_signal_bodies": sum(r["exact_signal"] for r in ok),
            "raw_equal_all_layers": sum(r["raw_equal"] for r in ok),
            "geometry_equal_all_layers": sum(r["count_ok"] and r["max_xor_um2"] <= tol_um2 for r in ok),
            "conductor_count_equal": sum(r["conductors_ok"] for r in ok),
            "chip_size_equal": sum(r["chip_ref_um"] == r["chip_new_um"] for r in ok),
            "max_xor_um2_by_layer": {
                k: max(r["layers"].get(k, {"xor_um2": 0.0})["xor_um2"] for r in ok)
                for k in sorted({k for r in ok for k in r["layers"]})
            },
            "export_style": dict(Counter(r.get("export_style") for r in ok if r.get("export_style"))),
            "median_seconds": sorted(r["seconds"] for r in rs)[len(rs) // 2],
        }
    return out


def _env() -> dict:
    """Library versions that change exported vertices (GEOS: exp11 drawn under 3.13, exp12 under 3.11)."""
    import gdstk
    import shapely

    try:
        import gdspy

        gdspy_version = gdspy.__version__
    except ImportError:
        gdspy_version = None
    return {
        "shapely": shapely.__version__,
        "geos": ".".join(map(str, shapely.geos_version)),
        "gdstk": gdstk.__version__,
        "gdspy": gdspy_version,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True, help="tl-squadds data_local directory")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--tol-um2", type=float, default=1e-6, help="xor area counted as equal geometry")
    ap.add_argument(
        "--star-style",
        choices=("auto", "original", "reexport", "reexport_gdspy"),
        default="auto",
        help="star export (squadds.components.export_fixups); auto: exp12 reexport_gdspy, exp11 original else reexport",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    tasks = []
    with tempfile.TemporaryDirectory() as tmp:
        for fam in args.families:
            items = collect(args.data_root, fam)
            for exp, folder in stratified_sample(fam, items, args.per_family, rng):
                tasks.append((fam, exp, str(folder), tmp, args.star_style))
        print(f"{len(tasks)} designs over {len(args.families)} families, {args.workers} workers", flush=True)
        t0 = time.time()
        with Pool(args.workers, maxtasksperchild=50) as pool:
            records = pool.map(check_one, tasks, chunksize=2)
        wall = time.time() - t0
    summary = summarize(records, args.tol_um2)
    for fam, s in summary.items():
        print(
            f"{fam:20s} n={s['n']:3d} err={s['errors']:2d} exact={s['exact_hash_all_layers']:3d} raw={s['raw_equal_all_layers']:3d} "
            f"signal_exact={s['exact_hash_signal_bodies']:3d} geom_eq={s['geometry_equal_all_layers']:3d} "
            f"conductors={s['conductor_count_equal']:3d} max_xor={s['max_xor_um2_by_layer']}"
        )
    print(f"wall {wall:.1f} s, {_env()}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "env": _env(), "wall_s": wall, "records": records}, indent=1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
