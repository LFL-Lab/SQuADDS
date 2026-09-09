"""Focused contract tests for the standalone Tutorial 24 predictive studies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tutorial24_predictive_studies.py"
SPEC = importlib.util.spec_from_file_location("tutorial24_predictive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_frame(per_family: int = 20) -> pd.DataFrame:
    rows = []
    for family_index, family in enumerate(MODULE.FAMILIES):
        for index in range(per_family):
            rows.append(
                {
                    "design_id": f"{family}:{index}",
                    "component_name": family,
                    "source_id": f"source:{family_index}:{index}",
                    "mutual_fF": 1.0 + family_index * 10 + index,
                    "ground_sum_fF": 2.0 + index,
                }
            )
    return pd.DataFrame(rows)


def test_group_split_has_no_design_leakage_and_is_repeatable():
    frame = synthetic_frame()
    eligible = np.flatnonzero(frame.component_name == MODULE.FAMILIES[0])
    first = MODULE.group_train_test_indices(frame, seed=8, eligible=eligible)
    second = MODULE.group_train_test_indices(frame, seed=8, eligible=eligible)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert set(frame.iloc[first[0]].design_id).isdisjoint(frame.iloc[first[1]].design_id)


def test_nested_subsets_share_one_order_and_have_exact_design_budgets():
    frame = synthetic_frame()
    pool = np.flatnonzero(frame.component_name == MODULE.FAMILIES[0])
    subsets = MODULE.nested_group_subsets(frame, pool, [1, 5, 10], seed=3)

    assert [len(subsets[size]) for size in (1, 5, 10)] == [1, 5, 10]
    assert set(subsets[1]).issubset(subsets[5])
    assert set(subsets[5]).issubset(subsets[10])


def test_family_balanced_pairs_have_equal_strata_and_no_self_pairs():
    frame = synthetic_frame()
    pairs = MODULE.family_balanced_pairs(frame.component_name, 17, seed=4)

    assert pairs.groupby("family_pair").size().nunique() == 1
    assert set(pairs.groupby("family_pair").size()) == {17}
    assert np.all(pairs.left.to_numpy() != pairs.right.to_numpy())
    assert pairs.family_pair.nunique() == 6


def test_native_options_exclude_pose_and_non_numeric_fields():
    options = {
        "d0": {
            "finger_width": "4um",
            "finger_count": 6,
            "orientation": "90deg",
            "pos_x": "100um",
            "nested": {"gap": "2.5um", "label": "alpha"},
        },
        "d1": {
            "finger_width": "5um",
            "finger_count": 7,
            "orientation": "0deg",
            "pos_x": "0um",
            "nested": {"gap": "3.5um", "label": "beta"},
        },
    }
    table = MODULE.native_option_table(options, ["d0", "d1"])

    assert list(table.columns) == ["design_id", "finger_count", "finger_width", "nested.gap"]
    assert table.finger_width.tolist() == [4.0, 5.0]


def test_transfer_uses_fixed_tests_and_nested_adaptation_sets():
    frame = synthetic_frame(15)
    rng = np.random.default_rng(5)
    representations = {"tiny": rng.normal(size=(len(frame), 6))}
    relations = MODULE.near_far_relations(frame, rng.normal(size=(len(frame), 8)))
    curves, assignments = MODULE.transfer_learning_study(
        frame,
        representations,
        relations,
        budgets=[1, 5, 10],
        repeats=1,
        test_fraction=0.2,
        features=8,
    )

    assert set(curves.target_labels) == {1, 5, 10}
    assert set(curves.method) == {
        "target_only",
        "pooled_source_target",
        "source_pretrained_residual_correction",
    }
    assert curves.groupby(["target_family", "repeat"]).test_group_hash.nunique().max() == 1
    for target, group in assignments.groupby("target_family"):
        test = set(group.loc[group.role == "test", "design_id"])
        pool = set(group.loc[group.role == "adaptation_pool", "design_id"])
        assert test.isdisjoint(pool), target
        assert sorted(group.loc[group.role == "adaptation_pool", "nested_rank"].astype(int)) == list(
            range(1, len(pool) + 1)
        )


def test_generalist_sample_count_is_controlled_across_diversity_levels():
    frame = synthetic_frame(24)
    matrix = np.random.default_rng(2).normal(size=(len(frame), 5))
    result = MODULE.generalist_growth_study(
        frame,
        {"tiny": matrix},
        budgets=[6, 12],
        repeats=1,
        test_fraction=0.2,
        features=8,
    )

    assert set(result.train_designs) == {6, 12}
    assert set(result.train_family_count) == {1, 2, 3}
    assert result.groupby(["train_families", "train_designs", "test_family"]).size().eq(1).all()


def test_regression_metrics_are_in_log_and_physical_units():
    expected = np.array([1.0, 2.0, 4.0, 8.0])
    scores = MODULE.regression_metrics(expected, np.log1p(expected))

    assert scores["r2_log1p"] == 1.0
    assert scores["rmse_fF"] < 1e-12
    assert scores["medae_fF"] < 1e-12
