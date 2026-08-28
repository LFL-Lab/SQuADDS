#!/usr/bin/env python
"""Build layout representations from the port-complete QMetal GDS dataset.

Four representations are produced, kept strictly separate from anything
published on Hugging Face:

``v0``
    ``static-shape-v0`` exactly as published.  Its role map recognizes ports
    only for ``GeneralizedCapNInterdigital``, so for the two regenerated
    families it ignores the new port markers entirely.  Rebuilding it therefore
    isolates the CapN conductor-geometry correction on its own.

``v0-ports``
    A clearly labelled local variant of v0 whose role map also recognizes the
    ordered port markers and the ``TransmonCross`` etch layer.  This is *not*
    ``static-shape-v0`` and must never be written to the published schema.

``v1-local``
    A local ``universal-geometry-v1`` fit on this two-family cohort.  Published
    v1 covers only ``GeneralizedCapNInterdigital`` and is fit on write, so its
    normalization statistics and variance-selected frequencies are not
    transferable; this build is a different space and is labelled as such.

``v2``
    ``universal-geometry-v2`` exactly as published.  It is the only shipped
    standard whose role map already consumes ports for every family, so it is
    where the ordered-port question can actually be asked.
"""

from __future__ import annotations

import os

for _variable in (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from squadds.layouts import canonical_design_id  # noqa: E402

DATABASE_REVISION = "0e25705f54c343fb96571ff15b6fd8375ca899aa"
DATABASE_FILES = {
    "TransmonCross": "qubit-TransmonCross-cap_matrix.json",
    "CapNInterdigitalTee": "coupler-CapNInterdigitalTee-cap_matrix.json",
}

#: Local v0 variant only.  Published v0 recognizes ports for the generalized
#: coupler alone; this map extends the same semantics to every family.
V0_PORTS_ROLE_MAP = {
    (1, 10): "conductor",
    (1, 11): "etch",
    (2, 0): "port",
    (3, 0): "port",
}
#: Etch but no ports, so the port effect can be separated from the etch effect.
#: Published v0 recognizes TransmonCross etch nowhere, so "v0-ports" alone
#: changes two things at once for that family.
V0_ETCH_ROLE_MAP = {
    (1, 10): "conductor",
    (1, 11): "etch",
}
ROLE_MAPS = {"v0-ports": V0_PORTS_ROLE_MAP, "v0-etch": V0_ETCH_ROLE_MAP}
#: v1 accepts an explicit layer-role mapping, so no patching is required.
V1_ROLE_MAP = dict(V0_PORTS_ROLE_MAP)

_STATE: dict = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def design_options(database_dir: Path) -> dict[str, dict]:
    options: dict[str, dict] = {}
    for component, filename in DATABASE_FILES.items():
        for row in json.loads((database_dir / filename).read_text()):
            table = row["design"]["design_options"]
            options[canonical_design_id(component, table)] = table
    return options


def _install_role_map(variant: str) -> None:
    """Patch v0's role map in this process only, for a labelled local variant."""
    from squadds.layouts import embeddings

    mapping = ROLE_MAPS[variant]

    def role(component_name: str, layer: int, datatype: int) -> str | None:
        return mapping.get((layer, datatype))

    embeddings._functional_role = role


def _initialize(root: str, options_path: str, variant: str | None) -> None:
    _STATE["root"] = Path(root)
    _STATE["options"] = json.loads(Path(options_path).read_text())
    if variant:
        _install_role_map(variant)


def _encode_v2(record: dict) -> dict | None:
    from squadds.layouts.geometry_v2 import encode

    options = _STATE["options"].get(record.get("design_id")) or {}
    try:
        vector, metadata = encode(_STATE["root"] / record["gds_path"], options, return_metadata=True)
    except Exception as error:  # noqa: BLE001
        return {"gds_path": record["gds_path"], "error": f"{type(error).__name__}: {error}"}
    return {
        "layout_id": record["layout_id"],
        "design_id": record.get("design_id"),
        "component_name": record["component_name"],
        "source_id": record.get("source_id"),
        "terminal_count": metadata["terminal_count"],
        "minimum_pair_gap_um": metadata["minimum_pair_gap_um"],
        "embedding": vector.astype(np.float32).tolist(),
    }


def _encode_v0(record: dict) -> dict | None:
    import math

    from squadds.layouts.embeddings import parameter_sum, rasterize_functional_shape

    options = _STATE["options"].get(record.get("design_id")) or {}
    try:
        bitmap, moments, _ = rasterize_functional_shape(
            _STATE["root"] / record["gds_path"], record["component_name"]
        )
    except Exception as error:  # noqa: BLE001
        return {"gds_path": record["gds_path"], "error": f"{type(error).__name__}: {error}"}
    return {
        "layout_id": record["layout_id"],
        "design_id": record.get("design_id"),
        "component_name": record["component_name"],
        "source_id": record.get("source_id"),
        "parameter_sum": parameter_sum(options) if options else math.nan,
        "moments": np.asarray(moments, dtype=np.float64).tolist(),
        "bitmap": bitmap.reshape(-1).astype(np.float32).tolist(),
    }


def assemble_v0(rows: list[dict]) -> pd.DataFrame:
    """Apply v0's published normalization to raw parts computed in workers."""
    import math

    parameters = np.asarray([row["parameter_sum"] for row in rows], dtype=np.float64)
    moments = np.vstack([np.asarray(row["moments"]) for row in rows])
    parameter_mean = float(np.nanmean(parameters))
    parameter_std = float(np.nanstd(parameters)) or 1.0
    moment_mean = moments.mean(axis=0)
    moment_std = moments.std(axis=0)
    moment_std[moment_std == 0] = 1.0

    records = []
    for index, row in enumerate(rows):
        parameter_block = np.asarray([math.tanh((parameters[index] - parameter_mean) / parameter_std)])
        moment_block = np.clip((moments[index] - moment_mean) / moment_std, -5.0, 5.0)
        norm = np.linalg.norm(moment_block)
        if norm:
            moment_block = moment_block / norm
        shape_block = np.asarray(row["bitmap"], dtype=np.float64)
        norm = np.linalg.norm(shape_block)
        if norm:
            shape_block = shape_block / norm
        vector = np.concatenate([parameter_block, moment_block, shape_block])
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        records.append(
            {
                "layout_id": row["layout_id"],
                "design_id": row["design_id"],
                "component_name": row["component_name"],
                "source_id": row["source_id"],
                "embedding": vector.astype(np.float32).tolist(),
            }
        )
    return pd.DataFrame(records)


def run_parallel(records, root, options_path, worker, workers, variant=None):
    rows, failures = [], []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=(str(root), str(options_path), variant),
    ) as pool:
        futures = [pool.submit(worker, record) for record in records]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            (failures if "error" in result else rows).append(result)
            if index % 500 == 0:
                log(f"  {index}/{len(records)} ({len(failures)} failed)")
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="layout-dataset directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("representation", choices=["v0", "v0-ports", "v0-etch", "v1-local", "v2"])
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    arguments = parser.parse_args()

    manifest = pd.read_parquet(arguments.dataset_root / "metadata/manifest.parquet")
    records = manifest.to_dict(orient="records")
    output = arguments.output_dir / arguments.representation
    output.mkdir(parents=True, exist_ok=True)
    options_path = output / "design-options.json"
    options_path.write_text(json.dumps(design_options(arguments.database_dir)))
    log(f"{arguments.representation}: {len(records)} manifest rows")

    if arguments.representation in {"v0", "v0-ports", "v0-etch"}:
        rows, failures = run_parallel(
            records,
            arguments.dataset_root,
            options_path,
            _encode_v0,
            arguments.workers,
            variant=arguments.representation if arguments.representation in ROLE_MAPS else None,
        )
        frame = assemble_v0(rows)
    elif arguments.representation == "v2":
        rows, failures = run_parallel(
            records, arguments.dataset_root, options_path, _encode_v2, arguments.workers
        )
        frame = pd.DataFrame(rows)
    else:
        from squadds.layouts import write_universal_embedding_dataset

        failures = []
        count, dimensions = write_universal_embedding_dataset(
            manifest,
            json.loads(options_path.read_text()),
            lambda record: arguments.dataset_root / record["gds_path"],
            output,
            layer_roles=V1_ROLE_MAP,
        )
        log(f"v1-local wrote {count} rows of {dimensions} dimensions")
        frame = pd.read_parquet(output / "metadata/universal-geometry-v1.parquet")
        frame = frame[["layout_id", "design_id", "component_name", "source_id", "embedding"]]

    frame = frame.sort_values("layout_id").reset_index(drop=True)
    frame.to_parquet(output / "embeddings.parquet", index=False)
    provenance = {
        "representation": arguments.representation,
        "published_standard": arguments.representation in {"v0", "v2"},
        "note": {
            "v0": "static-shape-v0 as published; ignores the new ports for these families",
            "v0-ports": "LOCAL VARIANT of v0 with a port-aware and etch-aware role map; not the published standard",
            "v0-etch": "LOCAL VARIANT of v0 with an etch-aware but port-blind role map; isolates etch from ports",
            "v1-local": "LOCAL v1 fit on this two-family cohort; not comparable to published universal-geometry-v1",
            "v2": "universal-geometry-v2 as published; consumes ordered ports for every family",
        }[arguments.representation],
        "dataset_root": str(arguments.dataset_root),
        "database_revision": DATABASE_REVISION,
        "rows": int(len(frame)),
        "dimensions": int(len(frame.iloc[0]["embedding"])),
        "failures": len(failures),
        "layer_role_map": {f"{k[0]}/{k[1]}": v for k, v in ROLE_MAPS.get(arguments.representation, V1_ROLE_MAP).items()}
        if arguments.representation in {"v0-ports", "v0-etch", "v1-local"}
        else "published default",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if failures:
        pd.DataFrame(failures).to_parquet(output / "failures.parquet", index=False)
    log(f"{arguments.representation}: wrote {len(frame)} rows x {provenance['dimensions']} dims ({len(failures)} failed)")


if __name__ == "__main__":
    main()
