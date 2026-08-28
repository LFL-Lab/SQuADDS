#!/usr/bin/env python
"""Check terminal discovery and ordering across every row of the sweep."""

from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

_ROOT: dict = {}


def _init(root: str) -> None:
    _ROOT["path"] = Path(root)


def _check(record: dict) -> dict:
    import shapely

    from squadds.layouts.geometry_v2 import _role_geometry, _terminals, read_layer_geometry

    path = _ROOT["path"] / record["gds_path"]
    try:
        grouped = _role_geometry(read_layer_geometry(path), None)
        conductor = shapely.union_all([s for _, s in grouped["conductor"]])
        with_ports = _terminals(conductor, grouped["port"])
        area_only = _terminals(conductor, [])
        flipped = (
            len(with_ports) == len(area_only) == 2
            and abs(with_ports[0].area - area_only[0].area) > 1e-9
        )
        return {
            "gds_path": record["gds_path"],
            "component_name": record["component_name"],
            "terminals": len(with_ports),
            "ports": len(grouped["port"]),
            "ordering_differs_from_area": bool(flipped),
            "ok": len(with_ports) == 2 and len(grouped["port"]) == 2,
        }
    except Exception as error:  # noqa: BLE001
        return {"gds_path": record["gds_path"], "component_name": record["component_name"],
                "terminals": -1, "ports": -1, "ordering_differs_from_area": False,
                "ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    arguments = parser.parse_args()

    manifest = pd.read_parquet(arguments.dataset_root / "metadata/manifest.parquet")
    records = manifest.to_dict(orient="records")
    rows = []
    with ProcessPoolExecutor(arguments.workers, initializer=_init,
                             initargs=(str(arguments.dataset_root),)) as pool:
        for future in as_completed([pool.submit(_check, r) for r in records]):
            rows.append(future.result())
    frame = pd.DataFrame(rows)
    frame.to_parquet(arguments.output.with_suffix(".parquet"), index=False)
    summary = {
        "rows": int(len(frame)),
        "all_two_terminals": bool((frame.terminals == 2).all()),
        "all_two_ports": bool((frame.ports == 2).all()),
        "failures": int((~frame.ok).sum()),
        "per_family": {
            component: {
                "rows": int(len(group)),
                "two_terminals": int((group.terminals == 2).sum()),
                "ordering_differs_from_area": int(group.ordering_differs_from_area.sum()),
            }
            for component, group in frame.groupby("component_name")
        },
    }
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
