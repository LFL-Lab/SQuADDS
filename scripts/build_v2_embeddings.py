#!/usr/bin/env python
"""Build universal-geometry-v2 embeddings from GDS layouts and design options.

Unlike the v0 and v1 builders this script performs no catalogue-wide fitting.
Every row is encoded independently, so the output of two separate runs on two
disjoint contributions can be concatenated and compared directly.
"""

from __future__ import annotations

import os

# These must be set before numpy is imported, in this process and in every
# re-imported child.  macOS Accelerate otherwise spawns a full thread pool per
# worker; the resulting oversubscription pinned throughput at roughly half a
# design per second no matter how many workers were requested, a 75x slowdown
# against the same call in the parent process.
for _variable in (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from squadds.layouts.geometry_v2 import (  # noqa: E402
    UNIVERSAL_V2_MODEL,
    UNIVERSAL_V2_SCHEMA_VERSION,
    V2_DIMENSIONS,
    encode,
    universal_v2_schema,
)
from squadds.layouts.manifest import canonical_design_id  # noqa: E402

_WORKER_STATE: dict[str, object] = {}


def _mapping(value: str) -> tuple[str, Path]:
    key, separator, raw_path = value.partition("=")
    if not separator or not key or not raw_path:
        raise argparse.ArgumentTypeError("Mappings must use key=/absolute/path format.")
    return key, Path(raw_path)


def _design_options(design_sources: list[tuple[str, Path]]) -> dict[str, dict]:
    options_by_id: dict[str, dict] = {}
    for component_name, source_path in design_sources:
        for row in json.loads(source_path.read_text()):
            options = row["design"]["design_options"]
            options_by_id[canonical_design_id(component_name, options)] = options
    return options_by_id


def _resolve(gds_root: Path, gds_path: str) -> Path:
    candidate = gds_root / gds_path
    if candidate.is_file():
        return candidate
    fallback = gds_root / Path(gds_path).name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing GDS artifact: {candidate}")


def _initialize(gds_root: str, options_path: str) -> None:
    _WORKER_STATE["gds_root"] = Path(gds_root)
    with open(options_path) as stream:
        _WORKER_STATE["options"] = json.load(stream)


def _encode_record(record: dict) -> dict | None:
    gds_root = _WORKER_STATE["gds_root"]
    options_by_id = _WORKER_STATE["options"]
    design_id = record.get("design_id")
    options = options_by_id.get(design_id) or {}
    try:
        vector, metadata = encode(_resolve(gds_root, record["gds_path"]), options, return_metadata=True)
    except Exception as error:  # noqa: BLE001 - a failed layout must not abort a 20k-row build
        return {"gds_path": record["gds_path"], "error": f"{type(error).__name__}: {error}"}
    return {
        "layout_id": record["layout_id"],
        "artifact_id": record["artifact_id"],
        "design_id": design_id,
        "component_name": record["component_name"],
        "source_id": record.get("source_id"),
        "campaign": record.get("campaign"),
        "embedding_version": "v2",
        "embedding_model": UNIVERSAL_V2_MODEL,
        "embedding_schema_version": UNIVERSAL_V2_SCHEMA_VERSION,
        "terminal_count": metadata["terminal_count"],
        "raster_pixel_um": metadata["raster_pixel_um"],
        "minimum_pair_gap_um": metadata["minimum_pair_gap_um"],
        "parameter_count": metadata["parameter_count"],
        "embedding": vector.astype(np.float32).tolist(),
    }


def build(
    manifest: pd.DataFrame,
    gds_root: Path,
    options_path: Path,
    output_dir: Path,
    workers: int,
) -> tuple[int, int]:
    records = manifest.to_dict(orient="records")
    rows: list[dict] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=(str(gds_root), str(options_path)),
    ) as pool:
        futures = [pool.submit(_encode_record, record) for record in records]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is None:
                continue
            (failures if "error" in result else rows).append(result)
            if index % 500 == 0:
                print(f"  encoded {index}/{len(records)} ({len(failures)} failed)", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    model_dir = output_dir / "models" / UNIVERSAL_V2_MODEL
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows).sort_values("layout_id").reset_index(drop=True)
    embedding_path = metadata_dir / "universal-geometry-v2.parquet"
    schema_path = model_dir / "schema.json"
    frame.to_parquet(embedding_path, index=False)
    schema_path.write_text(json.dumps(universal_v2_schema(), indent=2) + "\n")
    if failures:
        pd.DataFrame(failures).to_parquet(model_dir / "failures.parquet", index=False)

    release_files = {path.relative_to(output_dir).as_posix(): path for path in (embedding_path, schema_path)}
    release_manifest = {
        "model": UNIVERSAL_V2_MODEL,
        "embedding_schema_version": UNIVERSAL_V2_SCHEMA_VERSION,
        "component_name": sorted(frame["component_name"].unique().tolist()),
        "rows": len(frame),
        "dimensions": V2_DIMENSIONS,
        "fitted_on_catalogue": False,
        "layouts_without_downloadable_gds": len(failures),
        "files": {
            name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in release_files.items()
        },
    }
    (model_dir / "release-manifest.json").write_text(json.dumps(release_manifest, indent=2) + "\n")
    return len(frame), len(failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("gds_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--design-json", action="append", required=True, type=_mapping)
    parser.add_argument("--component-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--options-cache", type=Path, default=Path("design-options.json"))
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    if args.component_name:
        manifest = manifest.loc[manifest["component_name"] == args.component_name]
    if args.limit:
        manifest = manifest.head(args.limit)

    args.options_cache.parent.mkdir(parents=True, exist_ok=True)
    args.options_cache.write_text(json.dumps(_design_options(args.design_json)))

    count, failed = build(manifest, args.gds_root, args.options_cache, args.output_dir, args.workers)
    print(f"Wrote {count} universal-geometry-v2 embeddings with {V2_DIMENSIONS} dimensions ({failed} failed).")


if __name__ == "__main__":
    main()
