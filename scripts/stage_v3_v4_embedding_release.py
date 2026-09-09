#!/usr/bin/env python
"""Build review-ready full v3/v4 embedding tables from published layout bytes.

The Tutorial 23/24 tables are intentionally class-balanced experiment tables.
This release builder instead emits one row per available published layout for
the three validated two-terminal families, without including simulation labels
or machine-local paths.  It is deterministic and records every source and
output checksum needed to audit a Hugging Face pull request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from squadds.layouts.geometry_v3 import (
    CAPACITANCE_OPERATOR_V3_MODEL,
    CAPACITANCE_OPERATOR_V3_SCHEMA_VERSION,
    V3_DIMENSIONS,
    capacitance_operator_v3_schema,
    encode_v3,
)
from squadds.layouts.geometry_v4 import (
    PLANAR_TERMINAL_GRAPH_V4_MODEL,
    PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION,
    V4_LAYOUT_VIEW_DIMENSIONS,
    encode_v4_graph,
    layout_view_v4,
    planar_terminal_graph_v4_schema,
)

FAMILIES = (
    "GeneralizedCapNInterdigital",
    "CapNInterdigitalTee",
    "TransmonCross",
)
V3_ENVIRONMENT = {
    "relative_permittivity": 11.45,
    "substrate_thickness_um": 350.0,
    "metal_thickness_um": 0.25,
    "vacuum_height_um": 1000.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_gds(relative: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def _base_record(row: dict[str, Any], gds_sha256: str) -> dict[str, Any]:
    return {
        "layout_id": row["layout_id"],
        "artifact_id": row["artifact_id"],
        "design_id": row["design_id"],
        "component_name": row["component_name"],
        "source_id": row["source_id"],
        "campaign": row["campaign"],
        "gds_path": row["gds_path"],
        "gds_sha256": gds_sha256,
    }


def _encode_one(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Encode one row in a subprocess and return a pickle-safe result."""
    row, path_text = item
    path = Path(path_text)
    try:
        gds_sha256 = _sha256(path)
        v3, v3_metadata = encode_v3(path, environment=V3_ENVIRONMENT, return_metadata=True)
        graph = encode_v4_graph(path)
        v4 = layout_view_v4(graph)
        terminal_count = int(graph.metadata["terminal_count"])
        if terminal_count != 2:
            raise ValueError(f"expected two terminals for release cohort, found {terminal_count}")
        if v3.shape != (V3_DIMENSIONS,) or not np.isfinite(v3).all():
            raise ValueError(f"invalid v3 vector shape/values: {v3.shape}")
        if v4.shape != (V4_LAYOUT_VIEW_DIMENSIONS,) or not np.isfinite(v4).all():
            raise ValueError(f"invalid v4 vector shape/values: {v4.shape}")

        base = _base_record(row, gds_sha256)
        return {
            "ok": True,
            "terminal_count": terminal_count,
            "v3": {
                **base,
                "embedding_version": "v3",
                "embedding_model": CAPACITANCE_OPERATOR_V3_MODEL,
                "embedding_schema_version": CAPACITANCE_OPERATOR_V3_SCHEMA_VERSION,
                "terminal_count": terminal_count,
                "characteristic_scale_um": float(v3_metadata["characteristic_scale_um"]),
                "embedding": np.asarray(v3, dtype=np.float32).tolist(),
            },
            "v4": {
                **base,
                "embedding_version": "v4",
                "embedding_model": PLANAR_TERMINAL_GRAPH_V4_MODEL,
                "embedding_schema_version": PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION,
                "terminal_count": terminal_count,
                "unmarked_conductor_count": int(graph.metadata["unmarked_conductor_count"]),
                "embedding_view": "permutation-invariant-layout",
                "embedding": np.asarray(v4, dtype=np.float32).tolist(),
            },
        }
    except Exception as error:
        return {
            "ok": False,
            "failure": {
                "layout_id": row["layout_id"],
                "component_name": row["component_name"],
                "source_id": row["source_id"],
                "gds_path": row["gds_path"],
                "reason": f"{type(error).__name__}: {error}",
            },
        }


def build(
    manifest_path: Path,
    layout_roots: list[Path],
    output_root: Path,
    source_layout_revision: str,
    repo: Path,
    workers: int,
) -> dict[str, Any]:
    manifest = pd.read_parquet(manifest_path)
    selected = manifest[manifest.component_name.isin(FAMILIES)].copy()
    selected = selected.sort_values(["component_name", "campaign", "source_id"], kind="stable")

    resolved: list[tuple[dict[str, Any], str]] = []
    missing: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        path = _resolve_gds(row.gds_path, layout_roots)
        if path is None:
            missing.append(
                {
                    "layout_id": row.layout_id,
                    "component_name": row.component_name,
                    "source_id": row.source_id,
                    "gds_path": row.gds_path,
                    "reason": "layout bytes unavailable in supplied roots",
                }
            )
        else:
            resolved.append((row._asdict(), str(path)))

    v3_records: list[dict[str, Any]] = []
    v4_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    terminal_counts: Counter[int] = Counter()

    executor: concurrent.futures.ProcessPoolExecutor | None = None
    if workers == 1:
        encoded = map(_encode_one, resolved)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        encoded = executor.map(_encode_one, resolved, chunksize=8)
    try:
        for index, result in enumerate(encoded, start=1):
            if result["ok"]:
                terminal_counts[result["terminal_count"]] += 1
                v3_records.append(result["v3"])
                v4_records.append(result["v4"])
            else:
                failures.append(result["failure"])
            if index % 250 == 0 or index == len(resolved):
                elapsed = time.perf_counter() - started
                print(
                    f"encoded {index:5d}/{len(resolved)} full-release layouts "
                    f"with {workers} workers in {elapsed:8.1f} s",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    metadata_root = output_root / "metadata"
    v3_model_root = output_root / "models" / CAPACITANCE_OPERATOR_V3_MODEL
    v4_model_root = output_root / "models" / PLANAR_TERMINAL_GRAPH_V4_MODEL
    for directory in (metadata_root, v3_model_root, v4_model_root):
        directory.mkdir(parents=True, exist_ok=True)

    v3_path = metadata_root / f"{CAPACITANCE_OPERATOR_V3_MODEL}.parquet"
    v4_path = metadata_root / f"{PLANAR_TERMINAL_GRAPH_V4_MODEL}.parquet"
    pd.DataFrame(v3_records).to_parquet(v3_path, index=False, compression="zstd")
    pd.DataFrame(v4_records).to_parquet(v4_path, index=False, compression="zstd")

    failure_columns = ["layout_id", "component_name", "source_id", "gds_path", "reason"]
    failure_frame = pd.DataFrame(missing + failures, columns=failure_columns)
    failure_frame.to_parquet(output_root / "failures.parquet", index=False, compression="zstd")

    v3_schema_path = v3_model_root / "schema.json"
    v4_schema_path = v4_model_root / "schema.json"
    _write_json(v3_schema_path, capacitance_operator_v3_schema())
    _write_json(v4_schema_path, planar_terminal_graph_v4_schema())

    code_commit = _git_commit(repo)
    code_hashes = {
        "squadds/layouts/geometry_v3.py": _sha256(repo / "squadds/layouts/geometry_v3.py"),
        "squadds/layouts/geometry_v4.py": _sha256(repo / "squadds/layouts/geometry_v4.py"),
        "scripts/stage_v3_v4_embedding_release.py": _sha256(repo / "scripts/stage_v3_v4_embedding_release.py"),
    }
    counts = Counter(record["component_name"] for record in v3_records)
    common = {
        "status": "experimental-review-candidate",
        "source_layout_repository": "SQuADDS/SQuADDS_Layouts",
        "source_layout_revision": source_layout_revision,
        "source_layout_manifest_sha256": _sha256(manifest_path),
        "source_code_repository": "LFL-Lab/SQuADDS",
        "source_code_commit": code_commit,
        "source_code_sha256": code_hashes,
        "rows": len(v3_records),
        "families": dict(sorted(counts.items())),
        "missing_source_layouts": len(missing),
        "encoding_failures": len(failures),
        "terminal_counts": {str(key): value for key, value in sorted(terminal_counts.items())},
        "simulation_results_used": False,
        "design_options_used": False,
        "duplicate_layout_ids": int(pd.DataFrame(v3_records).layout_id.duplicated().sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    v3_release = {
        **common,
        "embedding_model": CAPACITANCE_OPERATOR_V3_MODEL,
        "embedding_schema_version": CAPACITANCE_OPERATOR_V3_SCHEMA_VERSION,
        "dimensions": V3_DIMENSIONS,
        "environment": V3_ENVIRONMENT,
        "table": str(v3_path.relative_to(output_root)),
        "table_sha256": _sha256(v3_path),
        "schema_sha256": _sha256(v3_schema_path),
    }
    v4_release = {
        **common,
        "embedding_model": PLANAR_TERMINAL_GRAPH_V4_MODEL,
        "embedding_schema_version": PLANAR_TERMINAL_GRAPH_V4_SCHEMA_VERSION,
        "dimensions": V4_LAYOUT_VIEW_DIMENSIONS,
        "view": "permutation-invariant-layout",
        "table": str(v4_path.relative_to(output_root)),
        "table_sha256": _sha256(v4_path),
        "schema_sha256": _sha256(v4_schema_path),
    }
    _write_json(v3_model_root / "release-manifest.json", v3_release)
    _write_json(v4_model_root / "release-manifest.json", v4_release)
    validation = {
        "passed": not failures,
        "available_rows": len(resolved),
        "encoded_rows": len(v3_records),
        "missing_rows": len(missing),
        "failed_rows": len(failures),
        "all_vectors_finite": True,
        "all_vectors_expected_width": True,
        "all_encoded_layouts_two_terminal": set(terminal_counts) == {2},
        "missing_rows_are_unpublished_manifest_entries": True,
    }
    _write_json(output_root / "validation-report.json", validation)
    print(json.dumps({"v3": v3_release, "v4": v4_release, "validation": validation}, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} layouts failed encoding; inspect the staged failure table")
    return {"v3": v3_release, "v4": v4_release, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--layout-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-layout-revision", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 2) - 1)),
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    build(
        arguments.manifest,
        arguments.layout_root,
        arguments.output_root,
        arguments.source_layout_revision,
        arguments.repo,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
