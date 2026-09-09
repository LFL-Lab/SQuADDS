#!/usr/bin/env python
"""Leakage-controlled predictive studies for Tutorial 24.

This runner deliberately keeps prediction downstream of the representation
acceptance tests.  It joins the balanced v2/v3/v4 cohort by ``design_id``, uses
group-disjoint splits everywhere, and writes tidy, inspectable results rather
than notebook state.  Capacitance is modelled as ``log1p(mutual_fF)``; reported
absolute errors are transformed back to fF.

The three historical families are not a substitute for a crossed acquisition:
solver settings, component family, and design-option vocabulary remain partly
confounded.  The manifest records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 24
FAMILIES = (
    "GeneralizedCapNInterdigital",
    "CapNInterdigitalTee",
    "TransmonCross",
)
V3_FIXED_STACK_DIMENSIONS = 240
POSE_FIELDS = {"pos_x", "pos_y", "orientation", "rotation", "x", "y"}
NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
SOURCE_FILES = {
    "GeneralizedCapNInterdigital": "coupler-GeneralizedCapNInterdigital-cap_matrix.json",
    "CapNInterdigitalTee": "coupler-CapNInterdigitalTee-cap_matrix.json",
    "TransmonCross": "qubit-TransmonCross-cap_matrix.json",
}


def _hash_strings(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, values))).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _stack(series: pd.Series) -> np.ndarray:
    matrix = np.vstack(series.to_numpy()).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite values.")
    return matrix


def load_cohort(v2_path: Path, v3_path: Path, v4_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Join the balanced cohort one-to-one and return aligned representations."""
    v2 = pd.read_parquet(v2_path).drop_duplicates("design_id", keep=False)
    v3 = pd.read_parquet(v3_path).drop_duplicates("design_id", keep=False)
    v4 = pd.read_parquet(v4_path).drop_duplicates("design_id", keep=False)
    required = {"design_id", "component_name"}
    for name, frame in (("v2", v2), ("v3", v3), ("v4", v4)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} table lacks columns: {sorted(missing)}")
    target_columns = ["design_id", "component_name", "source_id", "mutual_fF", "ground_sum_fF"]
    data = v4[target_columns].merge(
        v3[["design_id", "component_name", "mutual_fF"]],
        on=["design_id", "component_name"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_v3"),
    )
    if not np.allclose(data.mutual_fF, data.mutual_fF_v3, rtol=0, atol=1e-12):
        raise ValueError("v3 and v4 capacitance targets disagree.")
    data = data.drop(columns="mutual_fF_v3").merge(
        v2[["design_id", "component_name"]],
        on=["design_id", "component_name"],
        how="inner",
        validate="one_to_one",
    )
    data = data.sort_values(["component_name", "design_id"]).reset_index(drop=True)
    if set(data.component_name) != set(FAMILIES):
        raise ValueError(f"Expected exactly the three balanced families, found {sorted(data.component_name.unique())}.")
    v2_lookup = v2.set_index("design_id").loc[data.design_id]
    v3_lookup = v3.set_index("design_id").loc[data.design_id]
    v4_lookup = v4.set_index("design_id").loc[data.design_id]
    representations = {
        "v2_full": _stack(v2_lookup.embedding),
        "v3_fixed_stack_core": _stack(v3_lookup.embedding)[:, :V3_FIXED_STACK_DIMENSIONS],
        "v4_pair": _stack(v4_lookup.embedding_v4_pair),
    }
    return data, representations


def group_train_test_indices(
    frame: pd.DataFrame,
    *,
    seed: int,
    test_fraction: float = 0.2,
    eligible: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole design groups; no group can occur on both sides."""
    indices = np.arange(len(frame)) if eligible is None else np.asarray(eligible, dtype=int)
    groups = np.asarray(sorted(frame.iloc[indices].design_id.astype(str).unique()))
    if len(groups) < 2:
        raise ValueError("A grouped train/test split requires at least two design_id values.")
    shuffled = np.random.default_rng(seed).permutation(groups)
    count = min(len(groups) - 1, max(1, int(round(test_fraction * len(groups)))))
    test_groups = set(shuffled[:count])
    test = indices[frame.iloc[indices].design_id.astype(str).isin(test_groups).to_numpy()]
    train = indices[~frame.iloc[indices].design_id.astype(str).isin(test_groups).to_numpy()]
    if set(frame.iloc[train].design_id) & set(frame.iloc[test].design_id):
        raise RuntimeError("design_id leakage detected in split construction.")
    return train, test


def nested_group_subsets(
    frame: pd.DataFrame,
    pool: Sequence[int],
    budgets: Sequence[int],
    *,
    seed: int,
) -> dict[int, np.ndarray]:
    """Return nested groups from one permutation; requested labels mean designs."""
    pool = np.asarray(pool, dtype=int)
    order = nested_group_order(frame, pool, seed=seed)
    result: dict[int, np.ndarray] = {}
    for budget in sorted(set(map(int, budgets))):
        if budget < 1 or budget > len(order):
            continue
        chosen = set(order[:budget])
        result[budget] = pool[frame.iloc[pool].design_id.astype(str).isin(chosen).to_numpy()]
    return result


def nested_group_order(frame: pd.DataFrame, pool: Sequence[int], *, seed: int) -> np.ndarray:
    """Return the auditable group order from which every nested subset is cut."""
    pool = np.asarray(pool, dtype=int)
    groups = np.asarray(sorted(frame.iloc[pool].design_id.astype(str).unique()))
    return np.random.default_rng(seed).permutation(groups)


@dataclass
class RandomFeatureRidge:
    """Small deterministic nonlinear head with train-only preprocessing."""

    seed: int = SEED
    features: int = 64
    alpha: float = 1.0

    def fit(self, matrix: np.ndarray, target: np.ndarray) -> RandomFeatureRidge:
        matrix = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_ = np.where(self.scale_ > 1e-8, self.scale_, 1.0)
        standardized = np.clip((matrix - self.mean_) / self.scale_, -8.0, 8.0)
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(0.0, 1.0 / math.sqrt(max(matrix.shape[1], 1)), (matrix.shape[1], self.features))
        self.phase_ = rng.uniform(0.0, 2.0 * np.pi, self.features)
        design = self._design_from_standardized(standardized)
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return self

    def _design_from_standardized(self, standardized: np.ndarray) -> np.ndarray:
        nonlinear = np.sqrt(2.0 / self.features) * np.cos(standardized @ self.weights_ + self.phase_)
        return np.column_stack([np.ones(len(standardized)), nonlinear])

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        standardized = np.clip((np.asarray(matrix) - self.mean_) / self.scale_, -8.0, 8.0)
        return self._design_from_standardized(standardized) @ self.coefficients_


def regression_metrics(expected_fF: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    expected = np.asarray(expected_fF, dtype=np.float64)
    observed_log = np.log1p(expected)
    prediction = np.maximum(np.expm1(np.asarray(predicted_log, dtype=np.float64)), 0.0)
    denominator = float(np.sum((observed_log - observed_log.mean()) ** 2))
    r2 = float(1.0 - np.sum((observed_log - predicted_log) ** 2) / denominator) if denominator > 0 else float("nan")
    return {
        "r2_log1p": r2,
        "rmse_fF": float(np.sqrt(np.mean((expected - prediction) ** 2))),
        "medae_fF": float(np.median(np.abs(expected - prediction))),
    }


def _flatten_options(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_options(value[key], path)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            yield from _flatten_options(nested, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _numeric_option(name: str, value: Any) -> float | None:
    leaf = name.rsplit(".", 1)[-1].lower()
    if leaf in POSE_FIELDS or leaf.startswith("pos_"):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    if isinstance(value, str):
        match = NUMBER.match(value)
        return float(match.group(1)) if match else None
    return None


def native_option_table(options_by_design: Mapping[str, Mapping[str, Any]], design_ids: Sequence[str]) -> pd.DataFrame:
    """Build a family-local numeric table without pretending schemas align."""
    records = []
    for design_id in design_ids:
        options = options_by_design.get(str(design_id))
        if options is None:
            continue
        record: dict[str, Any] = {"design_id": str(design_id)}
        for name, value in _flatten_options(options):
            numeric = _numeric_option(name, value)
            if numeric is not None:
                record[name] = numeric
        records.append(record)
    table = pd.DataFrame(records)
    if table.empty:
        return pd.DataFrame(columns=["design_id"])
    numeric = [column for column in table if column != "design_id" and table[column].notna().mean() >= 0.95]
    return table[["design_id", *sorted(numeric)]]


def _canonical_design_id(component: str, options: Mapping[str, Any]) -> str:
    from squadds.layouts.manifest import canonical_design_id

    return canonical_design_id(component, options)


def load_source_options(component: str, path: Path) -> dict[str, Mapping[str, Any]]:
    rows = json.loads(path.read_text())
    return {
        _canonical_design_id(component, row["design"]["design_options"]): row["design"]["design_options"]
        for row in rows
    }


def family_baseline_study(
    data: pd.DataFrame,
    representations: Mapping[str, np.ndarray],
    native_tables: Mapping[str, pd.DataFrame],
    *,
    repeats: int,
    test_fraction: float,
    features: int,
) -> pd.DataFrame:
    records = []
    target = data.mutual_fF.to_numpy(float)
    for family in FAMILIES:
        eligible = np.flatnonzero(data.component_name.to_numpy() == family)
        native = native_tables.get(family, pd.DataFrame(columns=["design_id"]))
        native_lookup = native.set_index("design_id") if not native.empty else native
        family_representations = dict(representations)
        if len(native.columns) > 1 and set(data.iloc[eligible].design_id).issubset(native_lookup.index):
            family_representations["native_family_options"] = native_lookup.loc[data.iloc[eligible].design_id].to_numpy(
                float
            )
        for repeat in range(repeats):
            train, test = group_train_test_indices(
                data, seed=SEED + repeat, test_fraction=test_fraction, eligible=eligible
            )
            local_train = np.flatnonzero(np.isin(eligible, train))
            local_test = np.flatnonzero(np.isin(eligible, test))
            for name, matrix in family_representations.items():
                local = matrix[eligible] if len(matrix) == len(data) else matrix
                model = RandomFeatureRidge(SEED + repeat, features).fit(local[local_train], np.log1p(target[train]))
                scores = regression_metrics(target[test], model.predict(local[local_test]))
                records.append(
                    {
                        "family": family,
                        "repeat": repeat,
                        "representation": name,
                        "feature_dimensions": local.shape[1],
                        "train_designs": len(train),
                        "test_designs": len(test),
                        "train_group_hash": _hash_strings(data.iloc[train].design_id),
                        "test_group_hash": _hash_strings(data.iloc[test].design_id),
                        **scores,
                    }
                )
    return pd.DataFrame(records)


def robust_family_target(data: pd.DataFrame) -> np.ndarray:
    values = np.log1p(data.mutual_fF.to_numpy(float))
    normalized = np.zeros(len(data), dtype=float)
    for family in FAMILIES:
        rows = np.flatnonzero(data.component_name.to_numpy() == family)
        center = np.median(values[rows])
        mad = np.median(np.abs(values[rows] - center))
        normalized[rows] = (values[rows] - center) / max(1.4826 * mad, 1e-8)
    return normalized


def robust_scale(matrix: np.ndarray) -> np.ndarray:
    center = np.median(matrix, axis=0)
    scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
    keep = scale > 1e-8
    if not np.any(keep):
        return np.zeros((len(matrix), 1))
    return np.clip((matrix[:, keep] - center[keep]) / scale[keep], -20.0, 20.0)


def family_balanced_pairs(labels: Sequence[str], pairs_per_stratum: int, *, seed: int) -> pd.DataFrame:
    """Sample the same number of design pairs from each unordered family stratum."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for left_family, right_family in itertools.combinations_with_replacement(FAMILIES, 2):
        left = np.flatnonzero(labels == left_family)
        right = np.flatnonzero(labels == right_family)
        if not len(left) or not len(right):
            continue
        count = 0
        while count < pairs_per_stratum:
            first = int(rng.choice(left))
            second = int(rng.choice(right))
            if first == second:
                continue
            records.append(
                {
                    "left": first,
                    "right": second,
                    "family_pair": f"{left_family}|{right_family}",
                }
            )
            count += 1
    return pd.DataFrame(records)


def _distance_correlations(matrix: np.ndarray, target: np.ndarray, pairs: pd.DataFrame) -> list[dict[str, Any]]:
    standardized = robust_scale(matrix)
    left = pairs.left.to_numpy(int)
    right = pairs.right.to_numpy(int)
    distance = np.linalg.norm(standardized[left] - standardized[right], axis=1) / math.sqrt(standardized.shape[1])
    difference = np.abs(target[left] - target[right])
    records = []
    for stratum, rows in [
        ("all_family_balanced", np.arange(len(pairs))),
        *pairs.groupby("family_pair").indices.items(),
    ]:
        statistic = spearmanr(distance[rows], difference[rows]).statistic
        records.append({"family_pair": stratum, "pairs": len(rows), "spearman": float(statistic)})
    return records


def distance_capacitance_study(
    data: pd.DataFrame,
    representations: Mapping[str, np.ndarray],
    *,
    pairs_per_stratum: int,
    bootstrap: int,
) -> pd.DataFrame:
    labels = data.component_name.to_numpy()
    target = robust_family_target(data)
    base_pairs = family_balanced_pairs(labels, pairs_per_stratum, seed=SEED)
    records = []
    for name, matrix in representations.items():
        point = _distance_correlations(matrix, target, base_pairs)
        draws: dict[str, list[float]] = {row["family_pair"]: [] for row in point}
        for draw in range(bootstrap):
            rng = np.random.default_rng(SEED + 1009 * (draw + 1))
            sampled = np.concatenate(
                [
                    rng.choice(np.flatnonzero(labels == family), size=np.sum(labels == family), replace=True)
                    for family in FAMILIES
                ]
            )
            sampled_labels = labels[sampled]
            sampled_data = data.iloc[sampled].reset_index(drop=True)
            sampled_target = robust_family_target(sampled_data)
            sampled_pairs = family_balanced_pairs(sampled_labels, pairs_per_stratum, seed=SEED + 7919 * draw)
            for row in _distance_correlations(matrix[sampled], sampled_target, sampled_pairs):
                draws[row["family_pair"]].append(row["spearman"])
        for row in point:
            distribution = np.asarray(draws[row["family_pair"]], dtype=float)
            records.append(
                {
                    "representation": name,
                    **row,
                    "ci95_low": float(np.nanquantile(distribution, 0.025)) if len(distribution) else float("nan"),
                    "ci95_high": float(np.nanquantile(distribution, 0.975)) if len(distribution) else float("nan"),
                    "uncertainty_method": "family-stratified design-cluster bootstrap",
                    "target_normalization": "within-family robust z score of log1p(mutual_fF)",
                }
            )
    return pd.DataFrame(records)


def near_far_relations(data: pd.DataFrame, v4: np.ndarray) -> pd.DataFrame:
    standardized = robust_scale(v4)
    labels = data.component_name.to_numpy()
    centroids = {family: standardized[labels == family].mean(axis=0) for family in FAMILIES}
    records = []
    for target in FAMILIES:
        candidates = []
        for source in FAMILIES:
            if source != target:
                candidates.append((float(np.linalg.norm(centroids[target] - centroids[source])), source))
        candidates.sort()
        records.append(
            {
                "target_family": target,
                "source_family": candidates[0][1],
                "relation": "near",
                "centroid_distance": candidates[0][0],
            }
        )
        records.append(
            {
                "target_family": target,
                "source_family": candidates[-1][1],
                "relation": "far",
                "centroid_distance": candidates[-1][0],
            }
        )
    return pd.DataFrame(records)


def transfer_learning_study(
    data: pd.DataFrame,
    representations: Mapping[str, np.ndarray],
    relations: pd.DataFrame,
    *,
    budgets: Sequence[int],
    repeats: int,
    test_fraction: float,
    features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = data.component_name.to_numpy()
    y = np.log1p(data.mutual_fF.to_numpy(float))
    values_fF = data.mutual_fF.to_numpy(float)
    records, split_records = [], []
    for repeat in range(repeats):
        target_splits: dict[str, tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]] = {}
        for target in FAMILIES:
            eligible = np.flatnonzero(labels == target)
            pool, test = group_train_test_indices(
                data, seed=SEED + 101 * repeat, test_fraction=test_fraction, eligible=eligible
            )
            subset_seed = SEED + 10007 * repeat
            subsets = nested_group_subsets(data, pool, budgets, seed=subset_seed)
            target_splits[target] = pool, test, subsets
            ranks = {
                design_id: rank
                for rank, design_id in enumerate(nested_group_order(data, pool, seed=subset_seed), start=1)
            }
            for index in test:
                split_records.append(
                    {
                        "repeat": repeat,
                        "target_family": target,
                        "design_id": data.iloc[index].design_id,
                        "role": "test",
                        "nested_rank": np.nan,
                    }
                )
            for index in pool:
                split_records.append(
                    {
                        "repeat": repeat,
                        "target_family": target,
                        "design_id": data.iloc[index].design_id,
                        "role": "adaptation_pool",
                        "nested_rank": ranks.get(str(data.iloc[index].design_id), np.nan),
                    }
                )
        for relation in relations.itertuples(index=False):
            target, source = relation.target_family, relation.source_family
            _, test, subsets = target_splits[target]
            source_rows = np.flatnonzero(labels == source)
            for representation, matrix in representations.items():
                source_model = RandomFeatureRidge(SEED + repeat, features).fit(matrix[source_rows], y[source_rows])
                for budget, chosen in subsets.items():
                    models = {
                        "target_only": RandomFeatureRidge(SEED + repeat, features).fit(matrix[chosen], y[chosen]),
                        "pooled_source_target": RandomFeatureRidge(SEED + repeat, features).fit(
                            matrix[np.r_[source_rows, chosen]], y[np.r_[source_rows, chosen]]
                        ),
                    }
                    residual_model = RandomFeatureRidge(SEED + repeat, features).fit(
                        matrix[chosen], y[chosen] - source_model.predict(matrix[chosen])
                    )
                    predictions = {name: model.predict(matrix[test]) for name, model in models.items()}
                    predictions["source_pretrained_residual_correction"] = source_model.predict(
                        matrix[test]
                    ) + residual_model.predict(matrix[test])
                    for method, prediction in predictions.items():
                        records.append(
                            {
                                "target_family": target,
                                "source_family": source,
                                "relation": relation.relation,
                                "representation": representation,
                                "repeat": repeat,
                                "target_labels": budget,
                                "method": method,
                                "test_designs": len(test),
                                "test_group_hash": _hash_strings(data.iloc[test].design_id),
                                "adaptation_group_hash": _hash_strings(data.iloc[chosen].design_id),
                                **regression_metrics(values_fF[test], prediction),
                            }
                        )
    return pd.DataFrame(records), pd.DataFrame(split_records)


def _allocated_training_rows(
    data: pd.DataFrame,
    family_pools: Mapping[str, np.ndarray],
    train_families: Sequence[str],
    budget: int,
    *,
    seed: int,
) -> np.ndarray | None:
    base, remainder = divmod(int(budget), len(train_families))
    chosen = []
    for position, family in enumerate(sorted(train_families)):
        count = base + int(position < remainder)
        groups = np.asarray(sorted(data.iloc[family_pools[family]].design_id.astype(str).unique()))
        if count > len(groups):
            return None
        order = np.random.default_rng(seed + 997 * FAMILIES.index(family)).permutation(groups)
        keep = set(order[:count])
        pool = family_pools[family]
        chosen.extend(pool[data.iloc[pool].design_id.astype(str).isin(keep).to_numpy()])
    return np.asarray(chosen, dtype=int)


def generalist_growth_study(
    data: pd.DataFrame,
    representations: Mapping[str, np.ndarray],
    *,
    budgets: Sequence[int],
    repeats: int,
    test_fraction: float,
    features: int,
) -> pd.DataFrame:
    labels = data.component_name.to_numpy()
    y = np.log1p(data.mutual_fF.to_numpy(float))
    values_fF = data.mutual_fF.to_numpy(float)
    records = []
    family_sets = [subset for size in (1, 2, 3) for subset in itertools.combinations(FAMILIES, size)]
    for repeat in range(repeats):
        pools, tests = {}, {}
        for family in FAMILIES:
            eligible = np.flatnonzero(labels == family)
            pools[family], tests[family] = group_train_test_indices(
                data, seed=SEED + 313 * repeat, test_fraction=test_fraction, eligible=eligible
            )
        for families in family_sets:
            for budget in sorted(set(map(int, budgets))):
                train = _allocated_training_rows(data, pools, families, budget, seed=SEED + 1009 * repeat)
                if train is None or len(train) != budget:
                    continue
                for representation, matrix in representations.items():
                    model = RandomFeatureRidge(SEED + repeat, features).fit(matrix[train], y[train])
                    for test_family in FAMILIES:
                        test = tests[test_family]
                        records.append(
                            {
                                "representation": representation,
                                "repeat": repeat,
                                "train_family_count": len(families),
                                "train_families": "|".join(families),
                                "train_designs": len(train),
                                "test_family": test_family,
                                "test_family_seen_in_training": test_family in families,
                                "test_designs": len(test),
                                "train_group_hash": _hash_strings(data.iloc[train].design_id),
                                "test_group_hash": _hash_strings(data.iloc[test].design_id),
                                **regression_metrics(values_fF[test], model.predict(matrix[test])),
                            }
                        )
    return pd.DataFrame(records)


def _resolve_source_paths(overrides: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in overrides:
        family, separator, raw_path = item.partition("=")
        if not separator or family not in SOURCE_FILES:
            raise ValueError("--source-json must be FAMILY=/path/to/file.json for a known family.")
        paths[family] = Path(raw_path)
    cache = Path.home() / ".cache/huggingface/hub/datasets--SQuADDS--SQuADDS_DB/snapshots"
    for family, filename in SOURCE_FILES.items():
        if family in paths:
            continue
        candidates = sorted(cache.glob(f"*/{filename}"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            paths[family] = candidates[0]
    return paths


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    data, representations = load_cohort(arguments.v2, arguments.v3, arguments.v4)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    source_paths = _resolve_source_paths(arguments.source_json)
    native_tables = {}
    native_audit = []
    for family in FAMILIES:
        if family not in source_paths:
            native_audit.append({"family": family, "available": False, "reason": "source rows not found"})
            continue
        options = load_source_options(family, source_paths[family])
        table = native_option_table(options, data.loc[data.component_name == family, "design_id"])
        native_tables[family] = table
        native_audit.append(
            {
                "family": family,
                "available": len(table) == int(np.sum(data.component_name == family)),
                "matched_designs": len(table),
                "feature_dimensions": max(len(table.columns) - 1, 0),
                "source": str(source_paths[family]),
            }
        )

    family_results = family_baseline_study(
        data,
        representations,
        native_tables,
        repeats=arguments.repeats,
        test_fraction=arguments.test_fraction,
        features=arguments.rff_features,
    )
    distance_results = distance_capacitance_study(
        data,
        representations,
        pairs_per_stratum=arguments.pairs_per_stratum,
        bootstrap=arguments.bootstrap,
    )
    relations = near_far_relations(data, representations["v4_pair"])
    transfer, splits = transfer_learning_study(
        data,
        representations,
        relations,
        budgets=arguments.transfer_budgets,
        repeats=arguments.repeats,
        test_fraction=arguments.test_fraction,
        features=arguments.rff_features,
    )
    growth = generalist_growth_study(
        data,
        representations,
        budgets=arguments.growth_budgets,
        repeats=arguments.repeats,
        test_fraction=arguments.test_fraction,
        features=arguments.rff_features,
    )
    outputs = {
        "family_specific_baselines.csv": family_results,
        "distance_capacitance_correlation.csv": distance_results,
        "near_far_relations.csv": relations,
        "transfer_learning_curves.csv": transfer,
        "transfer_split_assignments.csv": splits,
        "generalist_growth_diversity.csv": growth,
        "native_feature_audit.csv": pd.DataFrame(native_audit),
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    input_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("v2", arguments.v2), ("v3", arguments.v3), ("v4", arguments.v4))
    }
    manifest = {
        "study": "tutorial24-predictive-studies-v1",
        "rows": len(data),
        "families": data.component_name.value_counts().sort_index().to_dict(),
        "representations": {name: matrix.shape[1] for name, matrix in representations.items()},
        "inputs_sha256": input_hashes,
        "seed": SEED,
        "repeats": arguments.repeats,
        "test_fraction": arguments.test_fraction,
        "transfer_budgets": list(arguments.transfer_budgets),
        "growth_budgets": list(arguments.growth_budgets),
        "pairs_per_family_stratum": arguments.pairs_per_stratum,
        "bootstrap_draws": arguments.bootstrap,
        "model": {"name": "random Fourier feature ridge", "features": arguments.rff_features, "alpha": 1.0},
        "outputs": {
            name: {"rows": len(frame), "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
            for name, frame in outputs.items()
        },
        "leakage_controls": [
            "all train/test partitions are disjoint in design_id",
            "target tests are fixed across label budgets, methods, source relations, and representations within a repeat",
            "target adaptation subsets are nested within repeat",
            "preprocessing and random-feature heads are fitted using allowed training rows only",
        ],
        "limitations": [
            "These three historical families confound topology with solver campaign and option vocabulary.",
            "v2_full includes its design-parameter block; this can expose family vocabulary and is reported without calling it geometry-only.",
            "Near/far source relations are descriptive centroid distances defined once in v4 geometry, not causal physical distance.",
            "Distance/capacitance correlation uses solver labels and is a downstream audit, not a representation invariance test.",
            "Family-specific native features are not aligned across families and are never used as a cross-family transfer baseline.",
            "Bootstrap intervals quantify design-sampling uncertainty in this cohort, not solver or fabrication uncertainty.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2", type=Path, default=Path.home() / ".cache/squadds/tutorial20/universal-geometry-v2-portcomplete.parquet"
    )
    parser.add_argument(
        "--v3", type=Path, default=Path("tutorials/runtime/tutorial23/capacitance-operator-v3-balanced.parquet")
    )
    parser.add_argument(
        "--v4", type=Path, default=Path("tutorials/runtime/tutorial24/planar-terminal-graph-v4-balanced.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("tutorials/runtime/tutorial24/predictive"))
    parser.add_argument(
        "--source-json", action="append", default=[], help="Override source rows as FAMILY=/path/file.json"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--rff-features", type=int, default=64)
    parser.add_argument("--pairs-per-stratum", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--transfer-budgets", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    parser.add_argument("--growth-budgets", type=int, nargs="+", default=[30, 100, 300, 600])
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_arguments())
    print(json.dumps({"rows": result["rows"], "outputs": result["outputs"]}, indent=2))
