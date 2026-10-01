#!/usr/bin/env python
"""Expanded capacitance-transfer study using the Tutorial 24/25 corpora.

The study joins the 45k-layout v4 study library to all available capacitance
labels: 20,062 Generalized IDC records across exports/exp7/exp8, 894 CapN Tee
records, and 1,933 unique Transmon records from the full v3/v4 release.  It also
audits the 16,379 raw exp8 per-pass matrices and measures support for the
unlabeled asymmetric IDC, concentric, star, and Firas campaigns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from squadds.layouts import canonical_design_id


SEED = 20261001
CAP_FLOOR_FF = 0.01
HUBER_DELTA = 0.05
TAU_LOG = 0.10
GIDC = "GeneralizedCapNInterdigital"
CAPN = "CapNInterdigitalTee"
TRANSMON = "TransmonCross"
TOPOLOGY_FAMILIES = (GIDC, CAPN, TRANSMON)
LABELED_GIDC_CAMPAIGNS = ("exports", "exp7_exports", "exp8_exports")
UNLABELED_CAMPAIGNS = (
    "exp9_exports",
    "exp10_exports",
    "exp11_exports",
    "firas_sims_exp1",
    "firas_sims_exp2",
)
RAW_SPECS = {
    GIDC: {
        "file": "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
        "a": "north",
        "b": "south",
    },
    CAPN: {
        "file": "coupler-CapNInterdigitalTee-cap_matrix.json",
        "a": "top",
        "b": "bottom",
    },
    TRANSMON: {
        "file": "qubit-TransmonCross-cap_matrix.json",
        "a": "cross",
        "b": "claw",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stack(series: pd.Series) -> np.ndarray:
    matrix = np.vstack(series.map(np.asarray).to_numpy()).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite representation values found.")
    return matrix


def huber(error: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta))


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    result = np.zeros(len(groups), dtype=np.float64)
    unique = np.unique(groups)
    for group in unique:
        mask = groups == group
        result[mask] = 1.0 / (len(unique) * mask.sum())
    return result * len(result)


def fit_offset(
    observed: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray | None = None,
) -> float:
    weights = np.ones(len(observed)) if groups is None else group_balanced_weights(groups)
    weights = weights / weights.sum()
    objective = lambda value: float(
        np.sum(weights * huber(observed - (base + float(value))))
    )
    return float(minimize_scalar(objective, bounds=(-10.0, 10.0), method="bounded").x)


@dataclass
class RandomFeatureRidge:
    seed: int
    features: int = 96
    alpha: float = 1.0

    def fit(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RandomFeatureRidge":
        matrix = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        weights = np.ones(len(matrix), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight)
        weights = weights * (len(weights) / weights.sum())
        self.mean_ = np.average(matrix, axis=0, weights=weights)
        centered = matrix - self.mean_
        self.scale_ = np.sqrt(np.average(centered**2, axis=0, weights=weights))
        self.scale_ = np.where(self.scale_ > 1e-8, self.scale_, 1.0)
        standardized = np.clip(centered / self.scale_, -8.0, 8.0)
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(
            0.0,
            1.0 / math.sqrt(max(matrix.shape[1], 1)),
            (matrix.shape[1], self.features),
        )
        self.phase_ = rng.uniform(0.0, 2.0 * np.pi, self.features)
        design = self._design(standardized)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)[:, None] if target.ndim == 2 else target * np.sqrt(weights)
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            weighted_design.T @ weighted_design + penalty,
            weighted_design.T @ weighted_target,
        )
        return self

    def _design(self, standardized: np.ndarray) -> np.ndarray:
        nonlinear = np.sqrt(2.0 / self.features) * np.cos(
            standardized @ self.weights_ + self.phase_
        )
        return np.column_stack([np.ones(len(standardized)), nonlinear])

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        standardized = np.clip((np.asarray(matrix) - self.mean_) / self.scale_, -8.0, 8.0)
        return self._design(standardized) @ self.coefficients_


def load_raw_databases(local_only: bool):
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    paths: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []
    for family, spec in RAW_SPECS.items():
        path = Path(
            hf_hub_download(
                "SQuADDS/SQuADDS_DB",
                spec["file"],
                repo_type="dataset",
                local_files_only=local_only,
            )
        )
        paths[family] = path
        a, b = spec["a"], spec["b"]
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in json.loads(path.read_text(encoding="utf-8")):
            result = row["sim_results"]
            design_id = canonical_design_id(family, row["design"]["design_options"])
            record: dict[str, Any] = {
                "design_id": design_id,
                "mutual_fF": abs(float(result[f"{a}_to_{b}"])),
                "ground_a_fF": abs(float(result[f"{a}_to_ground"])),
                "ground_b_fF": abs(float(result[f"{b}_to_ground"])),
                "diag_a_fF": abs(float(result[f"{a}_to_{a}"])),
                "diag_b_fF": abs(float(result[f"{b}_to_{b}"])),
                "diag_ground_fF": abs(float(result["ground_to_ground"])),
            }
            record["ground_sum_fF"] = record["ground_a_fF"] + record["ground_b_fF"]
            matrix = result.get("maxwell_matrix")
            if matrix:
                letters = (a[0].upper(), b[0].upper(), "G")
                value = np.asarray(
                    [[float(matrix[f"C_{first}_{second}"]) for second in letters] for first in letters],
                    dtype=np.float64,
                )
                eigenvalues = np.linalg.eigvalsh((value + value.T) / 2.0)
                record["matrix_json"] = json.dumps(value.tolist())
                for index, eigenvalue in enumerate(eigenvalues):
                    record[f"matrix_eigen_{index}_fF"] = float(eigenvalue)
            buckets.setdefault(design_id, []).append(record)
        family_labels: dict[str, dict[str, Any]] = {}
        duplicate_spreads = []
        for design_id, records in buckets.items():
            record = dict(records[0])
            numeric = [
                key
                for key in record
                if key != "design_id" and isinstance(record[key], (float, int))
            ]
            for key in numeric:
                values = np.asarray([item[key] for item in records], dtype=np.float64)
                record[key] = float(values.mean())
                if len(values) > 1:
                    duplicate_spreads.append(
                        float((values.max() - values.min()) / max(abs(values.mean()), CAP_FLOOR_FF))
                    )
            if len(records) > 1 and all("matrix_json" in item for item in records):
                matrices = np.stack([np.asarray(json.loads(item["matrix_json"])) for item in records])
                matrix = matrices.mean(axis=0)
                record["matrix_json"] = json.dumps(matrix.tolist())
                for index, eigenvalue in enumerate(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)):
                    record[f"matrix_eigen_{index}_fF"] = float(eigenvalue)
            record["duplicate_label_count"] = len(records)
            family_labels[design_id] = record
        summaries.append(
            {
                "family": family,
                "database_rows": sum(len(items) for items in buckets.values()),
                "unique_designs": len(buckets),
                "duplicate_rows": sum(len(items) - 1 for items in buckets.values()),
                "maximum_duplicate_relative_spread": max(duplicate_spreads, default=0.0),
                "full_matrix_designs": sum("matrix_json" in item for item in family_labels.values()),
            }
        )
        labels[family] = family_labels
    return labels, paths, pd.DataFrame(summaries)


def proxy_values(value: str) -> tuple[float, float, float, list[float]]:
    matrix = np.asarray(json.loads(value), dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 proxy matrix, found {matrix.shape}.")
    mutual = abs(float(matrix[0, 1]))
    ground_a = abs(float(matrix[0, 2]))
    ground_b = abs(float(matrix[1, 2]))
    eigenvalues = np.maximum(np.linalg.eigvalsh((matrix + matrix.T) / 2.0), 0.0)
    return mutual, ground_a + ground_b, abs(ground_a - ground_b), eigenvalues.tolist()


def proxy_values_from_embedding(value: Any) -> tuple[float, float, float, list[float]]:
    """Recover the v4 two-terminal proxy branches from the pooled layout view."""
    embedding = np.asarray(value, dtype=np.float64)
    if embedding.shape != (544,):
        raise ValueError(f"Expected a 544-D v4 layout view, found {embedding.shape}.")
    # Indices 82 and 146 pool log1p(node-to-ground proxy) over the two nodes.
    mean_ground_log = float(embedding[82])
    std_ground_log = abs(float(embedding[146]))
    ground_a, ground_b = np.expm1(
        [mean_ground_log + std_ground_log, mean_ground_log - std_ground_log]
    )
    ground_a, ground_b = max(float(ground_a), 0.0), max(float(ground_b), 0.0)
    # Global index 55 is log1p of the sum of all unordered proxy branches.
    total_branches = max(float(np.expm1(embedding[55])), 0.0)
    mutual = max(total_branches - ground_a - ground_b, 0.0)
    matrix = np.asarray(
        [
            [mutual + ground_a, -mutual, -ground_a],
            [-mutual, mutual + ground_b, -ground_b],
            [-ground_a, -ground_b, ground_a + ground_b],
        ]
    )
    eigenvalues = np.maximum(np.linalg.eigvalsh(matrix), 0.0)
    return mutual, ground_a + ground_b, abs(ground_a - ground_b), eigenvalues.tolist()


def configuration_path(source_path: str) -> Path:
    path = Path(source_path)
    return path.parent.parent / "json" / f"{path.stem}.json"


def frame_record(
    row: Any,
    label: dict[str, Any],
    domain: str,
    source_id: str,
) -> dict[str, Any]:
    proxy_mutual, proxy_ground, proxy_imbalance, proxy_eigen = proxy_values_from_embedding(
        row.embedding_v4_layout
    )
    return {
        "row_id": f"{domain}:{source_id}",
        "family": row.component_name,
        "domain": domain,
        "source_id": source_id,
        "design_id": label.get("design_id", ""),
        "physical_geometry_sha256": row.physical_geometry_sha256,
        "mutual_fF": label["mutual_fF"],
        "ground_a_fF": label.get("ground_a_fF", np.nan),
        "ground_b_fF": label.get("ground_b_fF", np.nan),
        "ground_sum_fF": label["ground_sum_fF"],
        "proxy_mutual_fF": proxy_mutual,
        "proxy_ground_sum_fF": proxy_ground,
        "proxy_ground_imbalance_fF": proxy_imbalance,
        "proxy_eigen_0_fF": proxy_eigen[0],
        "proxy_eigen_1_fF": proxy_eigen[1],
        "proxy_eigen_2_fF": proxy_eigen[2],
        "matrix_eigen_0_fF": label.get("matrix_eigen_0_fF", np.nan),
        "matrix_eigen_1_fF": label.get("matrix_eigen_1_fF", np.nan),
        "matrix_eigen_2_fF": label.get("matrix_eigen_2_fF", np.nan),
        "matrix_json": label.get("matrix_json"),
        "embedding": row.embedding_v4_layout,
    }


def frame_release_record(row: Any, label: dict[str, Any]) -> dict[str, Any]:
    proxy_mutual, proxy_ground, proxy_imbalance, proxy_eigen = proxy_values_from_embedding(
        row.embedding
    )
    source_id = str(row.source_id)
    return {
        "row_id": f"release:{source_id}",
        "family": row.component_name,
        "domain": f"release_{row.component_name}",
        "source_id": source_id,
        "design_id": row.design_id,
        "physical_geometry_sha256": row.gds_sha256,
        "mutual_fF": label["mutual_fF"],
        "ground_a_fF": label["ground_a_fF"],
        "ground_b_fF": label["ground_b_fF"],
        "ground_sum_fF": label["ground_sum_fF"],
        "proxy_mutual_fF": proxy_mutual,
        "proxy_ground_sum_fF": proxy_ground,
        "proxy_ground_imbalance_fF": proxy_imbalance,
        "proxy_eigen_0_fF": proxy_eigen[0],
        "proxy_eigen_1_fF": proxy_eigen[1],
        "proxy_eigen_2_fF": proxy_eigen[2],
        "matrix_eigen_0_fF": label.get("matrix_eigen_0_fF", np.nan),
        "matrix_eigen_1_fF": label.get("matrix_eigen_1_fF", np.nan),
        "matrix_eigen_2_fF": label.get("matrix_eigen_2_fF", np.nan),
        "matrix_json": label.get("matrix_json"),
        "embedding": row.embedding,
    }


def load_expanded_corpus(
    library_path: Path,
    old_balanced_path: Path,
    release_v4_path: Path,
    local_only: bool,
):
    library = pd.read_parquet(library_path)
    release = pd.read_parquet(release_v4_path)
    raw, raw_paths, raw_summary = load_raw_databases(local_only)
    coverage: list[dict[str, Any]] = []
    expanded_records: list[dict[str, Any]] = []

    custom = library[
        (library.component_name == GIDC)
        & (library.collection == "custom_sweeps")
        & library.campaign.isin(LABELED_GIDC_CAMPAIGNS)
    ]
    for row in custom.itertuples():
        config = configuration_path(row.source_path)
        options = json.loads(config.read_text(encoding="utf-8"))["cap_options"]
        design_id = canonical_design_id(GIDC, options)
        label = raw[GIDC].get(design_id)
        short_source = str(row.source_id).split("/")[-1]
        coverage.append(
            {
                "family": GIDC,
                "domain": row.campaign,
                "source_id": short_source,
                "design_id": design_id,
                "label_available": label is not None,
                "matrix_available": bool(label and "matrix_json" in label),
            }
        )
        if label:
            expanded_records.append(frame_record(row, label, row.campaign, short_source))

    representation_rows = []
    gidc_release = release[release.component_name == GIDC]
    gidc_expanded = pd.DataFrame(expanded_records)
    release_by_design = gidc_release.set_index("design_id")
    common = gidc_expanded[gidc_expanded.design_id.isin(release_by_design.index)]
    differences = []
    for row in common.itertuples():
        released = np.asarray(release_by_design.loc[row.design_id].embedding, dtype=np.float64)
        differences.append(float(np.max(np.abs(np.asarray(row.embedding) - released))))
    representation_rows.append(
        {
            "family": GIDC,
            "release_rows": len(gidc_release),
            "release_unique_layouts": gidc_release.layout_id.nunique(),
            "selected_labeled_rows": len(gidc_expanded),
            "overlap_designs": len(common),
            "maximum_overlap_embedding_difference": max(differences, default=np.nan),
        }
    )

    for family in (CAPN, TRANSMON):
        family_release = (
            release[release.component_name == family]
            .sort_values("source_id", kind="stable")
            .drop_duplicates("layout_id", keep="first")
        )
        missing = family_release[~family_release.design_id.isin(raw[family])]
        if len(missing):
            raise RuntimeError(f"{family} release has {len(missing)} layouts without labels.")
        for row in family_release.itertuples():
            label = raw[family][row.design_id]
            expanded_records.append(frame_release_record(row, label))
            coverage.append(
                {
                    "family": family,
                    "domain": f"release_{family}",
                    "source_id": row.source_id,
                    "design_id": row.design_id,
                    "label_available": True,
                    "matrix_available": "matrix_json" in label,
                }
            )
        representation_rows.append(
            {
                "family": family,
                "release_rows": int((release.component_name == family).sum()),
                "release_unique_layouts": family_release.layout_id.nunique(),
                "selected_labeled_rows": len(family_release),
                "overlap_designs": len(family_release),
                "maximum_overlap_embedding_difference": 0.0,
            }
        )

    old = pd.read_parquet(old_balanced_path)
    old_lookup = old.set_index(["component_name", "source_id"])
    historical_records: list[dict[str, Any]] = []
    historical = library[
        (library.campaign == "tutorial24_balanced")
        & library.component_name.isin(TOPOLOGY_FAMILIES)
    ]
    for row in historical.itertuples():
        matched = old_lookup.loc[(row.component_name, row.source_id)]
        label = {
            "mutual_fF": float(matched.mutual_fF),
            "ground_sum_fF": float(matched.ground_sum_fF),
        }
        record = frame_record(
            row,
            label,
            f"historical_{row.component_name}",
            str(row.source_id),
        )
        historical_records.append(record)

    expanded = pd.DataFrame(expanded_records).sort_values(
        ["family", "domain", "source_id"], kind="stable"
    ).reset_index(drop=True)
    historical_frame = pd.DataFrame(historical_records).sort_values(
        ["family", "source_id"], kind="stable"
    ).reset_index(drop=True)
    if expanded.physical_geometry_sha256.duplicated().any():
        raise RuntimeError("Expanded labeled corpus contains duplicate physical geometries.")
    return (
        expanded,
        historical_frame,
        library,
        pd.DataFrame(coverage),
        raw_paths,
        raw_summary,
        pd.DataFrame(representation_rows),
    )


def audit_exp8_matrices(
    results_root: Path,
    expanded: pd.DataFrame,
) -> pd.DataFrame:
    expected = expanded[expanded.domain == "exp8_exports"].set_index("source_id")
    rows = []
    matrix_paths = sorted(results_root.glob("q3d_cap_*/cap_matrix.csv"))
    for path in matrix_paths:
        source_id = path.parent.name.removeprefix("q3d_")
        with path.open(newline="", encoding="utf-8") as stream:
            passes = list(csv.DictReader(stream))
        final = passes[-1]
        previous = passes[-2] if len(passes) > 1 else final
        keys = {
            "mutual": "C(N,S) [pF]",
            "ground_a": "C(N,G) [pF]",
            "ground_b": "C(S,G) [pF]",
        }
        values = {name: abs(float(final[key])) * 1000.0 for name, key in keys.items()}
        previous_values = {name: abs(float(previous[key])) * 1000.0 for name, key in keys.items()}
        reciprocal = (
            ("C(N,S) [pF]", "C(S,N) [pF]"),
            ("C(N,G) [pF]", "C(G,N) [pF]"),
            ("C(S,G) [pF]", "C(G,S) [pF]"),
        )
        asymmetry = max(abs(float(final[left]) - float(final[right])) for left, right in reciprocal) * 1000.0
        relative_asymmetry = max(
            abs(float(final[left]) - float(final[right]))
            / max(0.5 * (abs(float(final[left])) + abs(float(final[right]))), CAP_FLOOR_FF / 1000.0)
            for left, right in reciprocal
        )
        matrix = np.asarray(
            [
                [float(final["C(N,N) [pF]"]), float(final["C(N,S) [pF]"]), float(final["C(N,G) [pF]"])],
                [float(final["C(S,N) [pF]"]), float(final["C(S,S) [pF]"]), float(final["C(S,G) [pF]"])],
                [float(final["C(G,N) [pF]"]), float(final["C(G,S) [pF]"]), float(final["C(G,G) [pF]"])],
            ]
        ) * 1000.0
        row_sum_fraction = np.max(np.abs(matrix.sum(axis=1)) / np.maximum(np.abs(np.diag(matrix)), CAP_FLOOR_FF))
        off_diagonal = matrix[~np.eye(3, dtype=bool)]
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
        reference = expected.loc[source_id]
        reference_matrix = np.asarray(json.loads(reference.matrix_json), dtype=np.float64)
        database_difference = float(np.max(np.abs(matrix - reference_matrix)))
        rows.append(
            {
                "source_id": source_id,
                "csv_sha256": sha256(path),
                "passes": len(passes),
                "final_pass": int(float(final["Pass []"])),
                "mutual_fF": values["mutual"],
                "ground_a_fF": values["ground_a"],
                "ground_b_fF": values["ground_b"],
                "maximum_reciprocal_asymmetry_fF": asymmetry,
                "maximum_reciprocal_relative_asymmetry": relative_asymmetry,
                "positive_off_diagonal_entries": int(np.sum(off_diagonal > 1e-12)),
                "minimum_eigenvalue_fF": float(eigenvalues.min()),
                "maximum_final_pass_relative_drift": max(
                    abs(values[name] - previous_values[name]) / max(values[name], CAP_FLOOR_FF)
                    for name in keys
                ),
                "maximum_row_sum_fraction": row_sum_fraction,
                "maximum_database_difference_fF": database_difference,
            }
        )
    return pd.DataFrame(rows).sort_values("source_id")


def database_matrix_audit(expanded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in expanded[expanded.matrix_json.notna()].itertuples():
        matrix = np.asarray(json.loads(row.matrix_json), dtype=np.float64)
        reciprocal = np.abs(matrix - matrix.T)
        scale = np.maximum(0.5 * (np.abs(matrix) + np.abs(matrix.T)), CAP_FLOOR_FF)
        off_diagonal = matrix[~np.eye(3, dtype=bool)]
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
        rows.append(
            {
                "domain": row.domain,
                "source_id": row.source_id,
                "maximum_reciprocal_relative_asymmetry": float(np.max(reciprocal / scale)),
                "positive_off_diagonal_entries": int(np.sum(off_diagonal > 1e-12)),
                "maximum_row_sum_fraction": float(
                    np.max(
                        np.abs(matrix.sum(axis=1))
                        / np.maximum(np.abs(np.diag(matrix)), CAP_FLOOR_FF)
                    )
                ),
                "minimum_eigenvalue_fF": float(eigenvalues.min()),
            }
        )
    return pd.DataFrame(rows)


def target_arrays(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "mutual":
        observed = frame.mutual_fF.to_numpy(float)
        proxy = frame.proxy_mutual_fF.to_numpy(float)
    elif target == "ground_sum":
        observed = frame.ground_sum_fF.to_numpy(float)
        proxy = frame.proxy_ground_sum_fF.to_numpy(float)
    else:
        raise ValueError(target)
    return np.log(np.maximum(observed, CAP_FLOOR_FF)), np.log(np.maximum(proxy, CAP_FLOOR_FF))


def regression_metrics(observed_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted_log) - np.asarray(observed_log)
    relative = np.abs(np.exp(np.clip(error, -20.0, 20.0)) - 1.0)
    observed = np.exp(observed_log)
    predicted = np.exp(np.clip(predicted_log, -20.0, 20.0))
    denominator = float(np.sum((observed_log - observed_log.mean()) ** 2))
    return {
        "n_test": len(observed_log),
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.quantile(relative, 0.90)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "bias_log": float(np.mean(error)),
        "slope_log": float(np.polyfit(observed_log, predicted_log, 1)[0]),
        "r2_log": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
        "medae_fF": float(np.median(np.abs(predicted - observed))),
        "rmse_fF": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }


def exact_stratified_split(proxy_log: np.ndarray, test_fraction: float, seed: int):
    ranks = pd.Series(proxy_log).rank(method="first")
    bins = pd.qcut(ranks, 10, labels=False, duplicates="drop").to_numpy()
    rng = np.random.default_rng(seed)
    desired = int(round(len(proxy_log) * test_fraction))
    members = {int(value): rng.permutation(np.flatnonzero(bins == value)) for value in np.unique(bins)}
    raw = {key: len(value) * test_fraction for key, value in members.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = desired - sum(counts.values())
    order = sorted(members, key=lambda key: (raw[key] - counts[key], -key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    test, train = [], []
    for key, values in members.items():
        test.extend(values[: counts[key]])
        train.extend(values[counts[key] :])
    return np.asarray(sorted(train)), np.asarray(sorted(test))


def fit_predict_methods(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    seeds: Iterable[int],
    weight_groups: bool = False,
):
    train_x, test_x = stack(train.embedding), stack(test.embedding)
    train_y, train_proxy = target_arrays(train, target)
    test_y, test_proxy = target_arrays(test, target)
    groups = train.family.to_numpy(str) if weight_groups else None
    sample_weight = group_balanced_weights(groups) if groups is not None else None
    offset = fit_offset(train_y, train_proxy, groups)
    output = [
        ("P0_raw_proxy", -1, test_proxy, float("nan")),
        ("P1_source_offset", -1, test_proxy + offset, offset),
    ]
    for seed in seeds:
        direct = RandomFeatureRidge(seed).fit(train_x, train_y, sample_weight)
        residual = RandomFeatureRidge(seed).fit(
            train_x, train_y - (train_proxy + offset), sample_weight
        )
        output.extend(
            [
                ("RFF_v4_layout_direct", seed, direct.predict(test_x), offset),
                (
                    "RFF_v4_layout_proxy_residual",
                    seed,
                    test_proxy + offset + residual.predict(test_x),
                    offset,
                ),
            ]
        )
    return test_y, output


def within_domain_study(expanded: pd.DataFrame, seeds: Iterable[int]) -> pd.DataFrame:
    rows = []
    for domain, frame in expanded.groupby("domain", sort=True):
        frame = frame.reset_index(drop=True)
        _, proxy = target_arrays(frame, "mutual")
        train_index, test_index = exact_stratified_split(proxy, 0.2, SEED + len(rows))
        train, test = frame.iloc[train_index], frame.iloc[test_index]
        for target in ("mutual", "ground_sum"):
            observed, predictions = fit_predict_methods(train, test, target, seeds)
            for method, seed, predicted, offset in predictions:
                rows.append(
                    {
                        "domain": domain,
                        "family": frame.family.iloc[0],
                        "target": target,
                        "method": method,
                        "seed": seed,
                        "source_offset": offset,
                        **regression_metrics(observed, predicted),
                    }
                )
    return pd.DataFrame(rows)


def cross_campaign_study(expanded: pd.DataFrame, seeds: Iterable[int]) -> pd.DataFrame:
    gidc = expanded[expanded.family == GIDC]
    rows = []
    for held in LABELED_GIDC_CAMPAIGNS:
        train = gidc[gidc.domain != held]
        test = gidc[gidc.domain == held]
        for target in ("mutual", "ground_sum"):
            observed, predictions = fit_predict_methods(train, test, target, seeds)
            for method, seed, predicted, offset in predictions:
                rows.append(
                    {
                        "held_campaign": held,
                        "target": target,
                        "method": method,
                        "seed": seed,
                        "source_offset": offset,
                        **regression_metrics(observed, predicted),
                    }
                )
    return pd.DataFrame(rows)


def topology_sources(
    expanded: pd.DataFrame,
    historical: pd.DataFrame,
    held_family: str,
):
    old = historical[historical.family != held_family].copy()
    expanded_source = expanded[expanded.family != held_family].copy()
    return {
        "old_balanced_2682": (old, False),
        "expanded_unweighted": (expanded_source, False),
        "expanded_family_balanced": (expanded_source, True),
    }


def topology_zero_shot(
    expanded: pd.DataFrame,
    historical: pd.DataFrame,
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for held in (CAPN, TRANSMON):
        test = expanded[expanded.family == held]
        for protocol, (train, weighted) in topology_sources(expanded, historical, held).items():
            for target in ("mutual", "ground_sum"):
                observed, predictions = fit_predict_methods(
                    train, test, target, seeds, weight_groups=weighted
                )
                for method, seed, predicted, offset in predictions:
                    rows.append(
                        {
                            "held_family": held,
                            "source_protocol": protocol,
                            "target": target,
                            "method": method,
                            "seed": seed,
                            "source_rows": len(train),
                            "source_offset": offset,
                            **regression_metrics(observed, predicted),
                        }
                    )
    return pd.DataFrame(rows)


def low_shot_topology(
    expanded: pd.DataFrame,
    historical: pd.DataFrame,
    seeds: Iterable[int],
    draws: int,
    budgets: Iterable[int],
):
    rows, assignments = [], []
    budgets = tuple(sorted(budgets))
    for held_index, held in enumerate((CAPN, TRANSMON)):
        target_frame = expanded[expanded.family == held].reset_index(drop=True)
        _, proxy = target_arrays(target_frame, "mutual")
        pool_index, test_index = exact_stratified_split(proxy, 0.5, SEED + held_index)
        pool, test = target_frame.iloc[pool_index], target_frame.iloc[test_index]
        for partition, part in (("shot_pool", pool), ("test", test)):
            for row_id in part.row_id:
                assignments.append(
                    {"held_family": held, "row_id": row_id, "partition": partition}
                )
        orders = [
            np.random.default_rng(SEED + 1000 * held_index + draw).permutation(len(pool))
            for draw in range(draws)
        ]
        for target in ("mutual", "ground_sum"):
            test_y, _ = target_arrays(test, target)
            pool_y, _ = target_arrays(pool, target)
            test_x, pool_x = stack(test.embedding), stack(pool.embedding)
            for protocol, (source, weighted) in topology_sources(expanded, historical, held).items():
                source_x = stack(source.embedding)
                source_y, _ = target_arrays(source, target)
                weights = group_balanced_weights(source.family.to_numpy(str)) if weighted else None
                for seed in seeds:
                    model = RandomFeatureRidge(seed).fit(source_x, source_y, weights)
                    base_test = model.predict(test_x)
                    base_pool = model.predict(pool_x)
                    rows.append(
                        {
                            "held_family": held,
                            "target": target,
                            "source_protocol": protocol,
                            "budget": 0,
                            "draw": 0,
                            "seed": seed,
                            "method": "source_zero_shot",
                            **regression_metrics(test_y, base_test),
                        }
                    )
                    residual_scale = max(float(np.std(source_y - model.predict(source_x), ddof=1)), 1e-3)
                    for draw, order in enumerate(orders):
                        for budget in budgets:
                            selected = order[:budget]
                            residual = pool_y[selected] - base_pool[selected]
                            shift = float(
                                residual.sum() / (budget + (residual_scale / TAU_LOG) ** 2)
                            )
                            rows.append(
                                {
                                    "held_family": held,
                                    "target": target,
                                    "source_protocol": protocol,
                                    "budget": budget,
                                    "draw": draw,
                                    "seed": seed,
                                    "method": "source_K_offset",
                                    **regression_metrics(test_y, base_test + shift),
                                }
                            )
            for seed in seeds:
                for draw, order in enumerate(orders):
                    for budget in budgets:
                        selected = order[:budget]
                        model = RandomFeatureRidge(seed).fit(pool_x[selected], pool_y[selected])
                        rows.append(
                            {
                                "held_family": held,
                                "target": target,
                                "source_protocol": "target_only",
                                "budget": budget,
                                "draw": draw,
                                "seed": seed,
                                "method": "target_only",
                                **regression_metrics(test_y, model.predict(test_x)),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(assignments)


def low_shot_campaign(
    expanded: pd.DataFrame,
    seeds: Iterable[int],
    draws: int,
    budgets: Iterable[int],
) -> pd.DataFrame:
    rows = []
    gidc = expanded[expanded.family == GIDC]
    budgets = tuple(sorted(budgets))
    for held_index, held in enumerate(LABELED_GIDC_CAMPAIGNS):
        target = gidc[gidc.domain == held].reset_index(drop=True)
        source = gidc[gidc.domain != held]
        target_y, proxy = target_arrays(target, "mutual")
        pool_index, test_index = exact_stratified_split(proxy, 0.5, SEED + 200 + held_index)
        pool_x, test_x = stack(target.embedding.iloc[pool_index]), stack(target.embedding.iloc[test_index])
        pool_y, test_y = target_y[pool_index], target_y[test_index]
        source_x, source_y = stack(source.embedding), target_arrays(source, "mutual")[0]
        orders = [
            np.random.default_rng(SEED + 3000 * held_index + draw).permutation(len(pool_index))
            for draw in range(draws)
        ]
        for seed in seeds:
            model = RandomFeatureRidge(seed).fit(source_x, source_y)
            base_test, base_pool = model.predict(test_x), model.predict(pool_x)
            residual_scale = max(float(np.std(source_y - model.predict(source_x), ddof=1)), 1e-3)
            rows.append(
                {
                    "held_campaign": held,
                    "budget": 0,
                    "draw": 0,
                    "seed": seed,
                    "method": "cross_campaign_zero_shot",
                    **regression_metrics(test_y, base_test),
                }
            )
            for draw, order in enumerate(orders):
                for budget in budgets:
                    selected = order[:budget]
                    residual = pool_y[selected] - base_pool[selected]
                    shift = float(
                        residual.sum() / (budget + (residual_scale / TAU_LOG) ** 2)
                    )
                    rows.append(
                        {
                            "held_campaign": held,
                            "budget": budget,
                            "draw": draw,
                            "seed": seed,
                            "method": "cross_campaign_K_offset",
                            **regression_metrics(test_y, base_test + shift),
                        }
                    )
                    target_model = RandomFeatureRidge(seed).fit(pool_x[selected], pool_y[selected])
                    rows.append(
                        {
                            "held_campaign": held,
                            "budget": budget,
                            "draw": draw,
                            "seed": seed,
                            "method": "target_only",
                            **regression_metrics(test_y, target_model.predict(test_x)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_metrics(
    rows: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    metrics = (
        "median_relative_error",
        "p90_relative_error",
        "bias_log",
        "slope_log",
        "r2_log",
        "medae_fF",
        "rmse_fF",
    )
    aggregations: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return rows.groupby(keys, as_index=False).agg(**aggregations).sort_values(keys)


def paired_topology_comparisons(low: pd.DataFrame, bootstrap: int = 10000) -> pd.DataFrame:
    nonzero = low[low.budget > 0].copy()
    nonzero["label"] = nonzero.source_protocol + ":" + nonzero.method
    index = ["held_family", "target", "budget", "draw", "seed"]
    wide = nonzero.pivot_table(
        index=index,
        columns="label",
        values="median_relative_error",
        aggfunc="first",
    ).reset_index()
    comparisons = {
        "expanded_balanced_vs_old": (
            "expanded_family_balanced:source_K_offset",
            "old_balanced_2682:source_K_offset",
        ),
        "expanded_unweighted_vs_old": (
            "expanded_unweighted:source_K_offset",
            "old_balanced_2682:source_K_offset",
        ),
        "expanded_balanced_vs_target_only": (
            "expanded_family_balanced:source_K_offset",
            "target_only:target_only",
        ),
    }
    rng = np.random.default_rng(SEED + 909)
    output = []
    for keys, group in wide.groupby(["held_family", "target", "budget"]):
        for name, (left, right) in comparisons.items():
            paired = group[["draw", left, right]].copy()
            paired["difference"] = paired[left] - paired[right]
            draw_values = paired.groupby("draw").difference.mean().to_numpy(float)
            samples = rng.choice(draw_values, (bootstrap, len(draw_values)), replace=True).mean(axis=1)
            output.append(
                {
                    "held_family": keys[0],
                    "target": keys[1],
                    "budget": keys[2],
                    "comparison": name,
                    "mean_paired_difference": float(draw_values.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "fraction_draws_left_better": float(np.mean(draw_values < 0)),
                }
            )
    return pd.DataFrame(output)


def robust_scale(matrix: np.ndarray):
    center = np.median(matrix, axis=0)
    spread = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
    active = spread > 1e-9
    return center, spread, active


def apply_scale(matrix: np.ndarray, center: np.ndarray, spread: np.ndarray, active: np.ndarray):
    return np.clip((matrix[:, active] - center[active]) / spread[active], -8.0, 8.0)


def sampled_pairs(left: np.ndarray, right: np.ndarray, count: int, seed: int, same: bool):
    rng = np.random.default_rng(seed)
    first = rng.choice(left, count, replace=True)
    second = rng.choice(right, count, replace=True)
    if same:
        collision = first == second
        while collision.any():
            second[collision] = rng.choice(right, int(collision.sum()), replace=True)
            collision = first == second
    return first, second


def similarity_study(expanded: pd.DataFrame, pairs: int) -> pd.DataFrame:
    balanced = pd.concat(
        [
            group.sample(min(len(group), 894), random_state=SEED)
            for _, group in expanded.groupby("domain", sort=True)
        ],
        ignore_index=True,
    )
    center, spread, active = robust_scale(stack(balanced.embedding))
    scaled = apply_scale(stack(expanded.embedding), center, spread, active)
    mutual, proxy_mutual = target_arrays(expanded, "mutual")
    ground, proxy_ground = target_arrays(expanded, "ground_sum")
    domains = expanded.domain.to_numpy(str)
    strata = [("within", domain, domain) for domain in sorted(expanded.domain.unique())]
    strata.extend(
        ("cross_campaign", first, second)
        for index, first in enumerate(LABELED_GIDC_CAMPAIGNS)
        for second in LABELED_GIDC_CAMPAIGNS[index + 1 :]
    )
    strata.extend(
        [
            ("cross_topology", "exp8_exports", f"release_{CAPN}"),
            ("cross_topology", "exp8_exports", f"release_{TRANSMON}"),
            ("cross_topology", f"release_{CAPN}", f"release_{TRANSMON}"),
        ]
    )
    rows = []
    for stratum_index, (kind, left_domain, right_domain) in enumerate(strata):
        left = np.flatnonzero(domains == left_domain)
        right = np.flatnonzero(domains == right_domain)
        first, second = sampled_pairs(
            left, right, pairs, SEED + stratum_index, left_domain == right_domain
        )
        embedding_distance = np.linalg.norm(scaled[first] - scaled[second], axis=1) / math.sqrt(active.sum())
        deltas = {
            "absolute_log_mutual_difference": np.abs(mutual[first] - mutual[second]),
            "absolute_log_ground_sum_difference": np.abs(ground[first] - ground[second]),
            "joint_log_response_distance": np.sqrt(
                (mutual[first] - mutual[second]) ** 2 + (ground[first] - ground[second]) ** 2
            ),
        }
        for outcome, delta in deltas.items():
            rows.append(
                {
                    "stratum": kind,
                    "domain_a": left_domain,
                    "domain_b": right_domain,
                    "relation": "v4_layout_distance",
                    "outcome": outcome,
                    "pairs": pairs,
                    "spearman_rho": float(spearmanr(embedding_distance, delta).statistic),
                }
            )
        for name, proxy, target in (
            ("proxy_mutual_delta", proxy_mutual, mutual),
            ("proxy_ground_delta", proxy_ground, ground),
        ):
            rows.append(
                {
                    "stratum": kind,
                    "domain_a": left_domain,
                    "domain_b": right_domain,
                    "relation": name,
                    "outcome": "matching_target_delta",
                    "pairs": pairs,
                    "spearman_rho": float(
                        spearmanr(
                            np.abs(proxy[first] - proxy[second]),
                            np.abs(target[first] - target[second]),
                        ).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)


def ood_support_study(
    expanded: pd.DataFrame,
    library: pd.DataFrame,
    dimensions: int = 32,
):
    balanced = pd.concat(
        [
            group.sample(min(len(group), 894), random_state=SEED)
            for _, group in expanded.groupby("domain", sort=True)
        ],
        ignore_index=True,
    )
    support_x = stack(expanded.embedding)
    center, spread, active = robust_scale(stack(balanced.embedding))
    support_scaled = apply_scale(support_x, center, spread, active)
    rng = np.random.default_rng(SEED)
    fit_index = rng.choice(len(support_scaled), min(4000, len(support_scaled)), replace=False)
    pca_center = support_scaled[fit_index].mean(axis=0)
    _, _, axes = np.linalg.svd(support_scaled[fit_index] - pca_center, full_matrices=False)
    axes = axes[:dimensions]
    support_projected = (support_scaled - pca_center) @ axes.T
    tree = cKDTree(support_projected)

    rows = []
    for row in library[library.campaign.isin(UNLABELED_CAMPAIGNS)].itertuples():
        vector = np.asarray(row.embedding_v4_layout, dtype=np.float64)[None, :]
        projected = (apply_scale(vector, center, spread, active) - pca_center) @ axes.T
        distance, neighbor = tree.query(projected[0], k=1)
        proxy_mutual, proxy_ground, _, _ = proxy_values(row.proxy_matrix_json)
        rows.append(
            {
                "campaign": row.campaign,
                "family": row.component_name,
                "source_id": row.source_id,
                "nearest_labeled_distance_32d": float(distance / math.sqrt(dimensions)),
                "nearest_labeled_family": expanded.family.iloc[int(neighbor)],
                "nearest_labeled_domain": expanded.domain.iloc[int(neighbor)],
                "proxy_mutual_fF": proxy_mutual,
                "proxy_ground_sum_fF": proxy_ground,
            }
        )
    row_frame = pd.DataFrame(rows)
    summary = (
        row_frame.groupby(["campaign", "family"], as_index=False)
        .agg(
            rows=("source_id", "size"),
            median_nearest_distance=("nearest_labeled_distance_32d", "median"),
            p90_nearest_distance=("nearest_labeled_distance_32d", lambda x: x.quantile(0.90)),
            p95_nearest_distance=("nearest_labeled_distance_32d", lambda x: x.quantile(0.95)),
            median_proxy_mutual_fF=("proxy_mutual_fF", "median"),
            median_proxy_ground_sum_fF=("proxy_ground_sum_fF", "median"),
        )
        .sort_values("median_nearest_distance")
    )
    neighbor_mix = (
        row_frame.groupby(["campaign", "nearest_labeled_family", "nearest_labeled_domain"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    return row_frame, summary, neighbor_mix


def matrix_spectrum_study(
    expanded: pd.DataFrame,
    seeds: Iterable[int],
    cross_campaign: bool,
) -> pd.DataFrame:
    target_columns = [f"matrix_eigen_{index}_fF" for index in range(3)]
    rows = []
    gidc = expanded[(expanded.family == GIDC) & expanded.matrix_json.notna()]
    for held_index, domain in enumerate(LABELED_GIDC_CAMPAIGNS):
        frame = gidc[gidc.domain == domain].reset_index(drop=True)
        if frame[target_columns].isna().any().any():
            raise ValueError(f"The {domain} matrix spectrum is incomplete.")
        observed = np.log(np.maximum(frame[target_columns].to_numpy(float), CAP_FLOOR_FF))
        if cross_campaign:
            train = gidc[gidc.domain != domain]
            train_x = stack(train.embedding)
            train_y = np.log(np.maximum(train[target_columns].to_numpy(float), CAP_FLOOR_FF))
            test_x, test_y = stack(frame.embedding), observed
        else:
            _, proxy = target_arrays(frame, "mutual")
            train_index, test_index = exact_stratified_split(proxy, 0.2, SEED + 808 + held_index)
            train_x, test_x = stack(frame.embedding.iloc[train_index]), stack(frame.embedding.iloc[test_index])
            train_y, test_y = observed[train_index], observed[test_index]
        for seed in seeds:
            model = RandomFeatureRidge(seed).fit(train_x, train_y)
            prediction = model.predict(test_x)
            for index in range(3):
                rows.append(
                    {
                        "held_campaign": domain,
                        "protocol": "cross_campaign" if cross_campaign else "within_campaign",
                        "eigenmode": index,
                        "seed": seed,
                        **regression_metrics(test_y[:, index], prediction[:, index]),
                    }
                )
    return pd.DataFrame(rows)


def dataset_summary(expanded: pd.DataFrame) -> pd.DataFrame:
    return (
        expanded.groupby(["family", "domain"], as_index=False)
        .agg(
            rows=("row_id", "size"),
            unique_geometries=("physical_geometry_sha256", "nunique"),
            mutual_median_fF=("mutual_fF", "median"),
            ground_sum_median_fF=("ground_sum_fF", "median"),
            proxy_mutual_median_fF=("proxy_mutual_fF", "median"),
        )
        .sort_values(["family", "domain"])
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--library",
        type=Path,
        default=Path("tutorials/runtime/study-library/study-library-v4.parquet"),
    )
    result.add_argument(
        "--old-balanced",
        type=Path,
        default=Path("tutorials/runtime/tutorial24/planar-terminal-graph-v4-balanced.parquet"),
    )
    result.add_argument(
        "--release-v4",
        type=Path,
        default=Path("../SQuADDS-hf-v34-release/metadata/planar-terminal-graph-v4.parquet"),
    )
    result.add_argument(
        "--exp8-results",
        type=Path,
        default=Path.home() / "Downloads/squadds_data/exports/squadds",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("tutorials/runtime/expanded-capacitance-transfer-v2"),
    )
    result.add_argument("--draws", type=int, default=20)
    result.add_argument("--seeds", type=int, default=5)
    result.add_argument("--pairs", type=int, default=5000)
    result.add_argument("--allow-downloads", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(SEED + 100 * index for index in range(args.seeds))
    (
        expanded,
        historical,
        library,
        coverage,
        raw_paths,
        raw_summary,
        representation_summary,
    ) = load_expanded_corpus(
        args.library, args.old_balanced, args.release_v4, not args.allow_downloads
    )
    dataset_summary(expanded).to_csv(args.output / "dataset_summary.csv", index=False)
    coverage.to_csv(args.output / "label_coverage.csv", index=False)
    raw_summary.to_csv(args.output / "raw_database_summary.csv", index=False)
    representation_summary.to_csv(args.output / "representation_coverage.csv", index=False)

    matrix_audit = audit_exp8_matrices(args.exp8_results, expanded)
    matrix_audit.to_csv(args.output / "exp8_matrix_audit.csv", index=False)
    matrix_summary = matrix_audit.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T
    matrix_summary.to_csv(args.output / "exp8_matrix_audit_summary.csv")
    database_audit = database_matrix_audit(expanded)
    database_audit.to_csv(args.output / "database_matrix_audit_rows.csv", index=False)
    database_audit.groupby("domain").agg(
        rows=("source_id", "size"),
        maximum_reciprocal_relative_asymmetry=("maximum_reciprocal_relative_asymmetry", "max"),
        positive_off_diagonal_entries=("positive_off_diagonal_entries", "sum"),
        median_row_sum_fraction=("maximum_row_sum_fraction", "median"),
        p95_row_sum_fraction=("maximum_row_sum_fraction", lambda values: values.quantile(0.95)),
        minimum_eigenvalue_fF=("minimum_eigenvalue_fF", "min"),
    ).reset_index().to_csv(args.output / "database_matrix_audit_summary.csv", index=False)

    within = within_domain_study(expanded, seeds)
    within.to_csv(args.output / "within_domain_predictive.csv", index=False)
    cross_campaign = cross_campaign_study(expanded, seeds)
    cross_campaign.to_csv(args.output / "cross_campaign_zero_shot.csv", index=False)
    topology = topology_zero_shot(expanded, historical, seeds)
    topology.to_csv(args.output / "topology_zero_shot.csv", index=False)

    low, assignments = low_shot_topology(
        expanded, historical, seeds, args.draws, (1, 3, 10, 30, 100)
    )
    low.to_csv(args.output / "topology_low_shot_metrics.csv", index=False)
    summarize_metrics(
        low,
        ["held_family", "target", "source_protocol", "budget", "method"],
    ).to_csv(args.output / "topology_low_shot_summary.csv", index=False)
    paired_topology_comparisons(low).to_csv(
        args.output / "topology_paired_comparisons.csv", index=False
    )
    assignments.to_csv(args.output / "topology_split_assignments.csv", index=False)

    campaign_low = low_shot_campaign(
        expanded, seeds, args.draws, (1, 3, 10, 30, 100)
    )
    campaign_low.to_csv(args.output / "campaign_low_shot_metrics.csv", index=False)
    summarize_metrics(
        campaign_low,
        ["held_campaign", "budget", "method"],
    ).to_csv(args.output / "campaign_low_shot_summary.csv", index=False)

    similarity_study(expanded, args.pairs).to_csv(
        args.output / "similarity_correlations.csv", index=False
    )
    ood_rows, ood_summary, neighbor_mix = ood_support_study(expanded, library)
    ood_rows.to_csv(args.output / "unlabeled_ood_support_rows.csv", index=False)
    ood_summary.to_csv(args.output / "unlabeled_ood_support_summary.csv", index=False)
    neighbor_mix.to_csv(args.output / "unlabeled_nearest_domain_mix.csv", index=False)

    matrix_spectrum_study(expanded, seeds, False).to_csv(
        args.output / "gidc_matrix_spectrum_within.csv", index=False
    )
    matrix_spectrum_study(expanded, seeds, True).to_csv(
        args.output / "gidc_matrix_spectrum_cross_campaign.csv", index=False
    )

    manifest = {
        "study": "expanded-capacitance-transfer-v2",
        "seed": SEED,
        "labeled_rows": len(expanded),
        "labeled_rows_by_domain": expanded.domain.value_counts().sort_index().to_dict(),
        "historical_comparator_rows": len(historical),
        "study_library_rows": len(library),
        "exp8_raw_matrices": len(matrix_audit),
        "exp8_raw_matrix_set_sha256": hashlib.sha256(
            "\n".join(
                f"{row.source_id}:{row.csv_sha256}" for row in matrix_audit.itertuples()
            ).encode("utf-8")
        ).hexdigest(),
        "protocol": {
            "targets": ["ln(mutual_fF)", "ln(ground_a_fF + ground_b_fF)"],
            "representation": "planar-terminal-graph-v4 permutation-invariant 544-D layout view",
            "canonical_proxy": "v4 graph proxy matrix stored per layout",
            "topology_targets": [CAPN, TRANSMON],
            "labeled_gidc_campaigns": list(LABELED_GIDC_CAMPAIGNS),
            "unlabeled_ood_campaigns": list(UNLABELED_CAMPAIGNS),
            "budgets": [0, 1, 3, 10, 30, 100],
            "draws": args.draws,
            "model_seeds": list(seeds),
            "held_target_test_fraction": 0.5,
            "within_domain_test_fraction": 0.2,
        },
        "input_sha256": {
            str(args.library): sha256(args.library),
            str(args.old_balanced): sha256(args.old_balanced),
            str(args.release_v4): sha256(args.release_v4),
            **{str(path): sha256(path) for path in raw_paths.values()},
        },
        "implementation_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Concentric, star, exp9, and Firas campaigns have v4 geometry/proxy data but no joined high-fidelity labels.",
            "Only two non-GIDC labeled topology families are available for strict topology transfer.",
            "The three historical topology families influenced representation development; topology LOFO remains exploratory.",
            "RFF ridge is a controlled nonlinear probe, not the proposed final graph neural network.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "labeled_rows": len(expanded),
                "exp8_matrices": len(matrix_audit),
                "files": len(list(args.output.iterdir())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
