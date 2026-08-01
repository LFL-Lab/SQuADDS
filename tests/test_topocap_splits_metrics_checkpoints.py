"""Leakage, variable-matrix metrics, and checkpoint integrity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _topocap_helpers import physical_matrix

from squadds.ml.topocap.checkpoints import (
    CheckpointCorruptionError,
    CheckpointMismatchError,
    build_experiment_fingerprint,
    checkpoint_matches,
    load_checkpoint,
    save_checkpoint,
)
from squadds.ml.topocap.metrics import (
    area_under_learning_curve,
    evaluate_maxwell_matrices,
    interval_calibration,
    maxwell_physical_diagnostics,
    negative_transfer_rate,
    paired_group_bootstrap,
    risk_coverage_curve,
)
from squadds.ml.topocap.splits import (
    GroupedSplit,
    capn_group_labels,
    generalized_ncap_group_labels,
    leave_one_group_out_splits,
    make_grouped_split,
    make_nested_target_split,
    resolved_support_schedule,
    validate_preprocessing_fit_indices,
)


def test_grouped_splits_hold_out_complete_domains_and_block_preprocessing_leakage():
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object)
    split = make_grouped_split(groups, test_groups=["d"], validation_groups=["c"], seed=7)

    assert split.train_idx.tolist() == [0, 1, 2, 3]
    assert split.validation_idx.tolist() == [4, 5]
    assert split.test_idx.tolist() == [6, 7]
    split.validate(groups)
    np.testing.assert_array_equal(
        validate_preprocessing_fit_indices(split, split.train_idx, require_all_train=True),
        split.train_idx,
    )
    with pytest.raises(ValueError, match="non-training"):
        validate_preprocessing_fit_indices(split, [0, 4])


def test_observed_group_leakage_is_detected_even_without_declared_group_labels():
    leaked = GroupedSplit(train_idx=[0], validation_idx=[], test_idx=[1], group_name="profile")
    with pytest.raises(ValueError, match="leakage"):
        leaked.validate(["same-profile", "same-profile"])


def test_nested_support_is_deterministic_and_never_touches_validation_or_test():
    groups = np.repeat(["a", "b", "c", "d"], 6)
    outer = make_grouped_split(groups, test_groups=["d"], validation_groups=["c"], seed=19)
    first = make_nested_target_split(outer, groups, support_fraction=0.25, seed=23)
    second = make_nested_target_split(outer, groups, support_fraction=0.25, seed=23)

    np.testing.assert_array_equal(first.fit_indices, second.fit_indices)
    assert set(first.fit_indices).isdisjoint(outer.validation_idx)
    assert set(first.fit_indices).isdisjoint(outer.test_idx)
    assert set(first.fit_indices) | set(first.unused_target_idx) == set(outer.train_idx)
    schedule = resolved_support_schedule(12, sizes=(0, 2, 20), fractions=(0.25, 1.0))
    assert [level.size for level in schedule] == [0, 2, 3, 12]


def test_logo_splits_and_geometry_labels_are_deterministic_and_unit_normalized():
    frame = pd.DataFrame(
        {
            "design_options": [
                {
                    "finger_count": count,
                    "finger_length": length,
                    "finger_width": "2um",
                    "finger_gap": "1um",
                    "finger_etch_radius": "0.2um",
                }
                for count, length in ((4, "20um"), (4, "0.02mm"), (5, "20um"))
            ],
            "source_campaign": ["exp6", "exp6", "exp7"],
        }
    )
    labels = generalized_ncap_group_labels(frame, mode="finger_profile")
    campaign = generalized_ncap_group_labels(frame, mode="campaign")
    capn = capn_group_labels(pd.DataFrame({"finger_count": [2, 2, 3]}))
    splits = leave_one_group_out_splits(campaign, validation="none")

    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
    assert campaign.tolist() == ["generalized:campaign:exp6"] * 2 + ["generalized:campaign:exp7"]
    assert capn[0] == capn[1] != capn[2]
    assert len(splits) == 2
    for split in splits:
        split.validate(campaign)


def test_leave_one_group_out_orders_numeric_validation_domains_numerically():
    groups = np.asarray([1, 2, 10, 1, 2, 10])
    splits = leave_one_group_out_splits(groups, validation="next")
    observed = {int(groups[split.test_idx[0]]): int(groups[split.validation_idx[0]]) for split in splits}

    assert observed == {1: 2, 2: 10, 10: 1}


def test_variable_node_metrics_cover_every_unique_matrix_entry_and_physicality():
    truth = [physical_matrix(node_count, scale=0.7 + node_count / 10) for node_count in range(2, 17)]
    prediction = [matrix.copy() for matrix in truth]
    report = evaluate_maxwell_matrices(truth, prediction, sample_ids=[f"n={n}" for n in range(2, 17)])

    assert report.aggregate["micro_mae_ff"] == pytest.approx(0.0)
    assert report.aggregate["macro_relative_frobenius"] == pytest.approx(0.0)
    assert report.aggregate["physical_valid_rate"] == pytest.approx(1.0)
    assert len(report.per_sample) == 15
    assert report.per_sample["node_count"].tolist() == list(range(2, 17))
    assert set(report.per_entry["i"]) == set(range(16))
    assert report.physical["valid"].all()

    nonphysical = truth[0].copy()
    nonphysical[0, 1] = nonphysical[1, 0] = abs(nonphysical[0, 1])
    assert not maxwell_physical_diagnostics(nonphysical).valid


def test_cluster_bootstrap_negative_transfer_and_selective_risk_are_reproducible():
    baseline = np.asarray([1.0, 1.2, 2.0, 2.2, 3.0, 3.2])
    transfer = np.asarray([0.8, 1.0, 1.8, 2.0, 3.2, 3.4])
    groups = ["a", "a", "b", "b", "c", "c"]
    first = paired_group_bootstrap(baseline, transfer, groups, n_bootstrap=300, seed=8)
    second = paired_group_bootstrap(baseline, transfer, groups, n_bootstrap=300, seed=8)

    assert first.to_dict() == second.to_dict()
    assert first.n_groups == 3
    assert first.negative_transfer_rate == pytest.approx(1 / 3)
    assert negative_transfer_rate(baseline, transfer, groups=groups) == pytest.approx(1 / 3)
    assert area_under_learning_curve([0, 1, 4], [1.0, 0.7, 0.4]) < 1.0

    risk = risk_coverage_curve([0.1, 0.2, 0.8, 1.0], [0.1, 0.2, 0.8, 1.0])
    assert risk.aurc == pytest.approx(risk.oracle_aurc)
    assert risk.excess_aurc == pytest.approx(0.0)


def test_interval_calibration_handles_variable_node_matrices():
    truth = [physical_matrix(2), physical_matrix(5)]
    lower = [matrix - 0.2 for matrix in truth]
    upper = [matrix + 0.2 for matrix in truth]
    report = interval_calibration(
        truth,
        lower,
        upper,
        nominal_coverage=0.9,
        groups=["small", "large"],
    )

    assert report.micro_coverage == pytest.approx(1.0)
    assert report.macro_coverage == pytest.approx(1.0)
    assert report.n_samples == 2
    assert report.n_elements == 3 + 15


def test_checkpoint_fingerprint_mismatch_and_byte_corruption_are_rejected(tmp_path):
    dataset = tmp_path / "dataset.json"
    gds = tmp_path / "layout.gds"
    dataset.write_text('[{"id": 1}]')
    gds.write_bytes(b"immutable-gds")
    common = {
        "dataset_files": {"simulation": dataset},
        "gds_files": {"layout": gds},
        "split": {"train": [0], "test": [1]},
        "feature_config": {"schema": "v1"},
        "runtime_identity": {"python": "test-runtime"},
        "code_version": "unit-test-v1",
    }
    fingerprint = build_experiment_fingerprint(model_config={"alpha": 1.0}, **common)
    mismatch = build_experiment_fingerprint(model_config={"alpha": 2.0}, **common)
    path = save_checkpoint(
        tmp_path / "model.checkpoint",
        {"weights": np.arange(4)},
        fingerprint,
        metadata={"fold": 2},
    )

    loaded = load_checkpoint(path, expected_fingerprint=fingerprint)
    np.testing.assert_array_equal(loaded.payload["weights"], np.arange(4))
    assert loaded.metadata == {"fold": 2}
    assert checkpoint_matches(path, fingerprint)
    assert not checkpoint_matches(path, mismatch)
    with pytest.raises(CheckpointMismatchError, match="mismatch"):
        load_checkpoint(path, expected_fingerprint=mismatch)

    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 0x01
    path.write_bytes(corrupted)
    with pytest.raises(CheckpointCorruptionError):
        load_checkpoint(path, expected_fingerprint=fingerprint)
    assert not checkpoint_matches(path, fingerprint)


def test_checkpoint_fingerprint_changes_when_dataset_bytes_change(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("first")
    kwargs = {
        "dataset_files": [dataset],
        "split": {"train": [0]},
        "feature_config": {"version": 1},
        "model_config": {"alpha": 1},
        "runtime_identity": {"runtime": "fixed"},
        "code_version": "fixed-code",
        "require_gds": False,
    }
    first = build_experiment_fingerprint(**kwargs)
    dataset.write_text("second")
    second = build_experiment_fingerprint(**kwargs)

    assert first.digest != second.digest
