#!/usr/bin/env python
"""Run paired, target-blind acceptance gates for universal-geometry-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > 1e-12, norms, 1.0)


def _control_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = sorted({name for row in frame["parameter_names"] for name in row})
    columns = {name: index for index, name in enumerate(names)}
    values = np.full((len(frame), len(names)), np.nan, dtype=np.float32)
    for row_index, (row_names, row_values) in enumerate(
        zip(frame["parameter_names"], frame["parameter_values"])
    ):
        for name, value in zip(row_names, row_values):
            values[row_index, columns[name]] = value
    means = np.nanmean(values, axis=0)
    standard_deviations = np.nanstd(values, axis=0)
    variable = standard_deviations > 1e-8
    standardized = (values[:, variable] - means[variable]) / standard_deviations[variable]
    return np.nan_to_num(standardized), list(np.asarray(names)[variable])


def _nearest(matrix: np.ndarray, queries: np.ndarray) -> np.ndarray:
    scores = matrix[queries] @ matrix.T
    scores[np.arange(len(queries)), queries] = -np.inf
    return np.argmax(scores, axis=1)


def _quality(
    matrix: np.ndarray,
    queries: np.ndarray,
    controls: np.ndarray,
    finger_count: np.ndarray,
    bitmaps: np.ndarray,
    capacitance: np.ndarray,
) -> dict[str, float]:
    neighbors = _nearest(matrix, queries)
    return {
        "finger_count_exact": float(np.mean(finger_count[queries] == finger_count[neighbors])),
        "finger_count_mae": float(np.mean(np.abs(finger_count[queries] - finger_count[neighbors]))),
        "parameter_distance": float(
            np.mean(np.linalg.norm(controls[queries] - controls[neighbors], axis=1))
        ),
        "shape_distance": float(
            np.mean(np.linalg.norm(bitmaps[queries] - bitmaps[neighbors], axis=1))
        ),
        "mutual_capacitance_mae_ff": float(
            np.mean(np.abs(capacitance[queries] - capacitance[neighbors]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v0_parquet", type=Path)
    parser.add_argument("v1_parquet", type=Path)
    parser.add_argument("database_json", type=Path)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--query-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1401)
    args = parser.parse_args()

    v0 = pd.read_parquet(args.v0_parquet).query(
        "component_name == 'GeneralizedCapNInterdigital'"
    )
    v1 = pd.read_parquet(args.v1_parquet).query(
        "component_name == 'GeneralizedCapNInterdigital'"
    )
    paired = v1.merge(
        v0[["layout_id", "embedding"]].rename(columns={"embedding": "embedding_v0"}),
        on="layout_id",
        validate="one_to_one",
    )
    if len(paired) < args.sample_size:
        raise ValueError(f"Requested {args.sample_size} rows from a {len(paired)}-row catalogue.")

    rng = np.random.default_rng(args.seed)
    paired = paired.iloc[rng.choice(len(paired), args.sample_size, replace=False)].reset_index(
        drop=True
    )
    v0_matrix = np.vstack(paired["embedding_v0"]).astype(np.float32)
    v1_matrix = np.vstack(paired["embedding"]).astype(np.float32)
    bitmaps = _unit_rows(v0_matrix[:, 11:])
    controls, control_names = _control_matrix(paired)
    finger_count = controls[:, control_names.index("finger_count")]
    rows = json.loads(args.database_json.read_text())
    capacitance_by_source = {
        row["notes"]["source_id"]: row["sim_results"]["north_to_south"] for row in rows
    }
    capacitance = np.asarray(
        [capacitance_by_source[source_id] for source_id in paired["source_id"]]
    )
    queries = np.linspace(0, len(paired) - 1, args.query_count, dtype=int)
    quality = {
        "v0": _quality(
            v0_matrix,
            queries,
            controls,
            finger_count,
            bitmaps,
            capacitance,
        ),
        "v1": _quality(
            v1_matrix,
            queries,
            controls,
            finger_count,
            bitmaps,
            capacitance,
        ),
    }

    left = rng.integers(0, len(paired), size=8000)
    right = rng.integers(0, len(paired), size=8000)
    for label, matrix in (("v0", v0_matrix), ("v1", v1_matrix)):
        similarities = np.sum(matrix[left] * matrix[right], axis=1)
        quality[label]["random_pair_similarity_std"] = float(np.std(similarities))
        quality[label]["random_pair_similarity_q05"] = float(np.quantile(similarities, 0.05))
        quality[label]["random_pair_similarity_q95"] = float(np.quantile(similarities, 0.95))

    gates = {
        "topology": quality["v1"]["finger_count_exact"]
        >= quality["v0"]["finger_count_exact"],
        "parameters": quality["v1"]["parameter_distance"]
        <= 0.90 * quality["v0"]["parameter_distance"],
        "shape": quality["v1"]["shape_distance"] <= 1.10 * quality["v0"]["shape_distance"],
        "physics_proxy": quality["v1"]["mutual_capacitance_mae_ff"]
        <= quality["v0"]["mutual_capacitance_mae_ff"],
        "dynamic_range": quality["v1"]["random_pair_similarity_std"]
        >= 0.60 * quality["v0"]["random_pair_similarity_std"],
    }
    result = {
        "sample_size": len(paired),
        "query_count": len(queries),
        "seed": args.seed,
        "simulation_targets_used_to_fit_embedding": False,
        "quality": quality,
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("universal-geometry-v1 failed one or more acceptance gates")


if __name__ == "__main__":
    main()
