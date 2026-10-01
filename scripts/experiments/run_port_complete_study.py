#!/usr/bin/env python
"""Prediction and transfer studies on the port-complete QMetal GDS dataset.

The design isolates three separable effects rather than attributing every
difference to retraining:

===========================  ====================================================
effect                       contrast
===========================  ====================================================
CapN conductor correction    published v0 vs rebuilt v0 on CapN.  v0's role map
                             ignores the new ports, so the only thing that moves
                             is the corrected CPW geometry.
TransmonCross ordered ports  published v2 vs rebuilt v2 on TransmonCross.  Its
                             conductor geometry is byte-identical between the
                             two releases, so the only thing that moves is the
                             ports.
Ports under v0               rebuilt v0 vs rebuilt v0-ports, geometry held fixed.
===========================  ====================================================
"""

from __future__ import annotations

import os

for _variable in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "6")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

from squadds.layouts import V0KernelFeatureProjector, canonical_design_id, compress_v0_embeddings  # noqa: E402

CONFIG = {
    "database_repository": "SQuADDS/SQuADDS_DB",
    "database_revision": "0e25705f54c343fb96571ff15b6fd8375ca899aa",
    "layout_source": "port-complete QMetal regeneration (codex/transmon-port-gds @ 6b2788a)",
    "primary_targets": {
        "TransmonCross": {"field": "cross_to_claw", "unit": "fF"},
        "CapNInterdigitalTee": {"field": "top_to_bottom", "unit": "fF"},
        "GeneralizedCapNInterdigital": {"field": "north_to_south", "unit": "fF"},
    },
    "split_policy": "repeated grouped holdout, groups = design_id, 30% test",
    "repeats": 12,
    "test_fraction": 0.30,
    "seed": 24,
    "alpha": 0.3,
    "kernel_dimensions": 128,
    "pooled_shape_size": 12,
    "learning_curve_fractions": [0.02, 0.05, 0.10, 0.25, 0.50, 1.00],
}
DATABASE_FILES = {
    "TransmonCross": "qubit-TransmonCross-cap_matrix.json",
    "CapNInterdigitalTee": "coupler-CapNInterdigitalTee-cap_matrix.json",
    "GeneralizedCapNInterdigital": "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
}
SEED = CONFIG["seed"]
ALPHA = CONFIG["alpha"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class RidgeBank:
    """One factorization per training set, reusable for any prior."""

    def __init__(self, features, targets, alpha=ALPHA):
        augmented = np.column_stack([np.ones(len(features)), features])
        self.gram = augmented.T @ augmented
        self.rhs = augmented.T @ targets
        penalty = np.eye(augmented.shape[1])
        penalty[0, 0] = 0.0
        self.factor = cho_factor(self.gram + alpha * penalty, lower=True)
        self.shape = (augmented.shape[1], targets.shape[1])

    def fit(self, prior=None):
        if prior is None:
            prior = np.zeros(self.shape)
        return prior + cho_solve(self.factor, self.rhs - self.gram @ prior)


def predict(weights, features):
    return np.column_stack([np.ones(len(features)), features]) @ weights


def scores_ff(expected, predicted):
    """Metrics in physical units; expected and predicted are fF."""
    residual = np.asarray(expected).reshape(-1) - np.asarray(predicted).reshape(-1)
    denominator = np.sum((expected - np.mean(expected)) ** 2)
    return {
        "rmse_fF": float(np.sqrt(np.mean(residual**2))),
        "median_abs_error_fF": float(np.median(np.abs(residual))),
        "mae_fF": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / max(float(denominator), 1e-12)),
    }


def load_targets(database_dir: Path) -> pd.DataFrame:
    records = []
    for component, filename in DATABASE_FILES.items():
        field = CONFIG["primary_targets"][component]["field"]
        for row in json.loads((database_dir / filename).read_text()):
            options = row["design"]["design_options"]
            records.append(
                {
                    "design_id": canonical_design_id(component, options),
                    "component_name": component,
                    "target_fF": abs(float(row["sim_results"][field])),
                }
            )
    frame = pd.DataFrame(records)
    return frame.drop_duplicates("design_id").reset_index(drop=True)


def load_embeddings(path: Path, compress: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
    if compress:
        matrix = compress_v0_embeddings(matrix, CONFIG["pooled_shape_size"]).astype(np.float32)
    frame = frame[["design_id", "component_name"]].copy()
    frame["row"] = range(len(frame))
    return frame, matrix


def load_published_v0(path: Path, keep: set[str]):
    parquet = pq.ParquetFile(path)
    identifiers, blocks = [], []
    for batch in parquet.iter_batches(batch_size=256, columns=["design_id", "component_name", "embedding"]):
        chunk = batch.to_pandas()
        chunk = chunk[chunk.design_id.isin(keep)]
        if chunk.empty:
            continue
        matrix = np.vstack(chunk["embedding"].to_numpy()).astype(np.float32)
        blocks.append(compress_v0_embeddings(matrix, CONFIG["pooled_shape_size"]).astype(np.float32))
        identifiers.extend(chunk["design_id"].tolist())
    frame = pd.DataFrame({"design_id": identifiers})
    frame["row"] = range(len(frame))
    return frame.drop_duplicates("design_id"), np.vstack(blocks)


def grouped_splits(groups: np.ndarray, repeats: int, test_fraction: float):
    """Repeated holdout that never lets one design_id straddle train and test."""
    unique = np.unique(groups)
    splits = []
    for repeat in range(repeats):
        rng = np.random.default_rng(SEED + 101 * repeat)
        order = rng.permutation(unique)
        cut = max(1, int(round(len(order) * test_fraction)))
        held = set(order[:cut].tolist())
        mask = np.array([group in held for group in groups])
        splits.append((np.flatnonzero(~mask), np.flatnonzero(mask)))
    return splits


def within_family(name, matrix, table, component, out_records):
    rows = table.index[table.component_name == component].to_numpy()
    if len(rows) == 0:
        return
    features_raw = matrix[table.loc[rows, "row"].to_numpy()].astype(np.float64)
    projector = V0KernelFeatureProjector(kernel_dimensions=CONFIG["kernel_dimensions"], random_seed=SEED)
    features = projector.fit_transform_compact(features_raw)
    y = table.loc[rows, "target_fF"].to_numpy(float).reshape(-1, 1)
    groups = table.loc[rows, "design_id"].to_numpy()
    for repeat, (train, test) in enumerate(grouped_splits(groups, CONFIG["repeats"], CONFIG["test_fraction"])):
        for fraction in CONFIG["learning_curve_fractions"]:
            size = max(4, int(round(fraction * len(train))))
            chosen = train[:size]
            weights = RidgeBank(features[chosen], y[chosen]).fit()
            metrics = scores_ff(y[test].reshape(-1), predict(weights, features[test]).reshape(-1))
            out_records.append(
                {
                    "study": "within-family",
                    "representation": name,
                    "component_name": component,
                    "repeat": repeat,
                    "fraction": fraction,
                    "train_rows": size,
                    "test_rows": len(test),
                    **metrics,
                }
            )


def cross_family(name, source_matrix, source_table, target_matrix, target_table, out_records):
    """GeneralizedCapNInterdigital source, CapNInterdigitalTee target."""
    source_rows = source_table.index.to_numpy()
    target_rows = target_table.index.to_numpy()
    combined = np.vstack(
        [source_matrix[source_table.loc[source_rows, "row"].to_numpy()],
         target_matrix[target_table.loc[target_rows, "row"].to_numpy()]]
    ).astype(np.float64)
    projector = V0KernelFeatureProjector(kernel_dimensions=CONFIG["kernel_dimensions"], random_seed=SEED)
    projector.fit_compact(combined[: len(source_rows)])
    features = projector.transform_compact(combined)
    y = np.log1p(
        np.concatenate(
            [source_table.loc[source_rows, "target_fF"].to_numpy(float),
             target_table.loc[target_rows, "target_fF"].to_numpy(float)]
        )
    ).reshape(-1, 1)
    source_index = np.arange(len(source_rows))
    target_index = np.arange(len(source_rows), len(source_rows) + len(target_rows))
    foundation = RidgeBank(features[source_index], y[source_index]).fit()
    groups = target_table.loc[target_rows, "design_id"].to_numpy()

    for repeat, (train, test) in enumerate(grouped_splits(groups, CONFIG["repeats"], CONFIG["test_fraction"])):
        train, test = target_index[train], target_index[test]
        expected = np.expm1(y[test].reshape(-1))
        out_records.append(
            {
                "study": "cross-family", "representation": name, "method": "source-only (zero-shot)",
                "repeat": repeat, "fraction": 0.0, "train_rows": 0,
                **scores_ff(expected, np.expm1(predict(foundation, features[test]).reshape(-1))),
            }
        )
        for fraction in CONFIG["learning_curve_fractions"]:
            size = max(4, int(round(fraction * len(train))))
            chosen = train[:size]
            bank = RidgeBank(features[chosen], y[chosen])
            pooled_rows = np.concatenate([source_index, chosen])
            for method, weights in (
                ("target-only", bank.fit()),
                ("transfer (source prior)", bank.fit(foundation)),
                ("pooled", RidgeBank(features[pooled_rows], y[pooled_rows]).fit()),
            ):
                out_records.append(
                    {
                        "study": "cross-family", "representation": name, "method": method,
                        "repeat": repeat, "fraction": fraction, "train_rows": size,
                        **scores_ff(expected, np.expm1(predict(weights, features[test]).reshape(-1))),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--published-revision", default="main")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the published tables through the hub rather than accepting a path:
    # several stale snapshots of static-embedding-v0 are cached locally, and one
    # of them holds 4,577 rows instead of 24,106, which silently drops entire
    # component families from the old-versus-new comparison.
    published_v0_path = Path(
        hf_hub_download(
            "SQuADDS/SQuADDS_Layout_Embeddings",
            "metadata/static-embedding-v0.parquet",
            repo_type="dataset",
            revision=arguments.published_revision,
        )
    )
    published_v2_path = Path(
        hf_hub_download(
            "SQuADDS/SQuADDS_Layout_Embeddings",
            "metadata/universal-geometry-v2.parquet",
            repo_type="dataset",
            revision=arguments.published_revision,
        )
    )
    resolved_revision = published_v0_path.parent.parent.name
    CONFIG["published_embedding_revision"] = resolved_revision
    log(f"published tables resolved at revision {resolved_revision}")
    expected_v0_rows = pq.ParquetFile(published_v0_path).metadata.num_rows
    if expected_v0_rows < 24_000:
        raise SystemExit(f"published v0 has only {expected_v0_rows} rows; refusing to use a stale snapshot")

    targets = load_targets(arguments.database_dir)
    log(f"targets: {len(targets)} unique design_id rows")

    representations: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for name, compress in (("v0", True), ("v0-ports", True), ("v0-etch", True), ("v1-local", False), ("v2", False)):
        path = arguments.embedding_dir / name / "embeddings.parquet"
        if path.is_file():
            representations[f"new/{name}"] = load_embeddings(path, compress)
            log(f"loaded new/{name}: {representations[f'new/{name}'][1].shape}")

    published_v2 = pd.read_parquet(published_v2_path)
    published_v2 = published_v2[published_v2.component_name.isin(DATABASE_FILES)].drop_duplicates("design_id")
    matrix = np.vstack(published_v2["embedding"].to_numpy()).astype(np.float32)
    frame = published_v2[["design_id", "component_name"]].copy()
    frame["row"] = range(len(frame))
    representations["old/v2"] = (frame, matrix)
    log(f"loaded old/v2 (published, portless for the two regenerated families): {matrix.shape}")

    old_v0_frame, old_v0_matrix = load_published_v0(published_v0_path, set(targets.design_id))
    old_v0_frame = old_v0_frame.merge(targets[["design_id", "component_name"]], on="design_id")
    representations["old/v0"] = (old_v0_frame.reset_index(drop=True), old_v0_matrix)
    log(f"loaded old/v0 (published): {old_v0_matrix.shape}")

    records: list[dict] = []
    for name, (frame, matrix) in representations.items():
        table = frame.merge(targets[["design_id", "target_fF"]], on="design_id").reset_index(drop=True)
        for component in ("TransmonCross", "CapNInterdigitalTee"):
            within_family(name, matrix, table, component, records)
        log(f"  within-family done: {name}")

        source = table[table.component_name == "GeneralizedCapNInterdigital"].reset_index(drop=True)
        target = table[table.component_name == "CapNInterdigitalTee"].reset_index(drop=True)
        if len(source) and len(target):
            cross_family(name, matrix, source, matrix, target, records)
            log(f"  cross-family done: {name}")

    # universal-geometry-v2 consults no catalogue statistics, so the published
    # GeneralizedCapNInterdigital rows and the freshly built port-complete CapN
    # rows already occupy the same space and can be used as one transfer problem
    # without rebuilding either side.  No fit-on-write representation allows this.
    if "new/v2" in representations:
        old_frame, old_matrix = representations["old/v2"]
        old_table = old_frame.merge(targets[["design_id", "target_fF"]], on="design_id").reset_index(drop=True)
        source = old_table[old_table.component_name == "GeneralizedCapNInterdigital"].reset_index(drop=True)
        new_frame, new_matrix = representations["new/v2"]
        new_table = new_frame.merge(targets[["design_id", "target_fF"]], on="design_id").reset_index(drop=True)
        target = new_table[new_table.component_name == "CapNInterdigitalTee"].reset_index(drop=True)
        cross_family("mixed/v2 published-source + port-complete-target", old_matrix, source, new_matrix, target, records)
        log("  cross-family done: mixed/v2 (published Generalized source, port-complete CapN target)")

    frame = pd.DataFrame(records)
    frame.to_parquet(arguments.output_dir / "raw_metrics.parquet", index=False)
    (arguments.output_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    within = frame.query("study == 'within-family'")
    summary = within.groupby(["component_name", "representation", "fraction"], as_index=False).agg(
        rmse_fF=("rmse_fF", "mean"), rmse_sd=("rmse_fF", "std"),
        median_abs_error_fF=("median_abs_error_fF", "mean"), mae_fF=("mae_fF", "mean"),
        r2=("r2", "mean"), r2_sd=("r2", "std"), train_rows=("train_rows", "mean"),
    )
    summary.to_parquet(arguments.output_dir / "within_family_summary.parquet", index=False)
    print()
    print("=== within-family, full training pool")
    print(summary.query("fraction == 1.0").drop(columns=["fraction"]).round(4).to_string(index=False))
    log("saved")


if __name__ == "__main__":
    main()
