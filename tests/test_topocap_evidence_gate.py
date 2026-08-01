"""Support-only evidence gating for physical, permutation-safe transfer."""

from __future__ import annotations

import numpy as np
import pytest
from _topocap_helpers import synthetic_view_graph

from squadds.ml.topocap.adaptation import EBRAConfig
from squadds.ml.topocap.evidence_gate import (
    EvidenceGateConfig,
    EvidenceGatedTopoCap,
    _cross_validation_folds,
)
from squadds.ml.topocap.model import TopoCapConfig, TopoCapFoundationModel
from squadds.ml.topocap.targets import maxwell_diagnostics
from squadds.ml.topocap.views import build_active_geometry_view, build_topology_control_view


@pytest.fixture(scope="module")
def source_graphs():
    graphs = []
    for index in range(30):
        graphs.append(
            synthetic_view_graph(
                2 + index % 15,
                active_count=3.0 + index % 8,
                active_length_um=12.0 + 0.8 * index,
                active_width_um=1.0 + 0.1 * (index % 6),
                active_gap_um=0.5 + 0.08 * (index % 5),
                geometry_signal=0.2 + 0.03 * index,
                target_kind="control",
            )
        )
    return tuple(graphs)


@pytest.fixture(scope="module")
def control_foundation(source_graphs):
    return TopoCapFoundationModel(TopoCapConfig(random_feature_dimensions=0, ridge_alpha=0.1, random_seed=31)).fit(
        [build_topology_control_view(graph) for graph in source_graphs]
    )


def make_gate(control_foundation, **gate_overrides):
    return EvidenceGatedTopoCap(
        control_foundation,
        build_topology_control_view,
        build_active_geometry_view,
        adapter_config=EBRAConfig(prior_precision=0.1),
        specialist_config=TopoCapConfig(random_feature_dimensions=0, ridge_alpha=0.01, random_seed=37),
        gate_config=EvidenceGateConfig(**gate_overrides),
    )


def assert_physical_and_permutation_equivariant(gate, graph):
    order = np.random.default_rng(500 + graph.node_count).permutation(graph.node_count)
    original = gate.predict(graph)
    permuted = gate.predict(graph.reorder_nodes(order))

    assert maxwell_diagnostics(original.matrix).is_physical
    assert maxwell_diagnostics(original.mean_matrix).is_physical
    np.testing.assert_allclose(
        permuted.matrix,
        original.matrix[np.ix_(order, order)],
        rtol=3e-10,
        atol=3e-10,
    )


def test_zero_support_selects_foundation_and_predicts_every_node_count(control_foundation):
    gate = make_gate(control_foundation, minimum_specialist_support=8).fit([])
    probes = [synthetic_view_graph(node_count, active_count=5 + node_count % 3) for node_count in range(2, 17)]

    assert gate.choice == "foundation"
    assert gate.evidence_.support_size == 0
    assert gate.evidence_.fold_count == 0
    assert gate.adapter_ is None and gate.specialist_ is None
    predictions = gate.predict_many(probes)
    assert len(predictions) == 15
    for graph, prediction in zip(probes, predictions):
        assert prediction.matrix.shape == (graph.node_count, graph.node_count)
        assert maxwell_diagnostics(prediction.matrix).is_physical
    assert_physical_and_permutation_equivariant(gate, probes[-1])


def test_low_support_selects_transfer_without_model_selection(control_foundation):
    support = [
        synthetic_view_graph(
            3 + index,
            family="CapNInterdigitalTee",
            active_count=4 + index,
            active_length_um=16 + index,
            target_kind="control",
        )
        for index in range(4)
    ]
    gate = make_gate(control_foundation, minimum_specialist_support=8).fit(support)

    assert gate.choice == "transfer"
    assert gate.evidence_.support_size == 4
    assert gate.evidence_.transfer_cv_log_mae is None
    assert gate.evidence_.specialist_cv_log_mae is None
    assert gate.adapter_ is not None and gate.specialist_ is None
    assert_physical_and_permutation_equivariant(gate, support[-1])


def test_enough_support_can_select_geometry_specialist(control_foundation):
    support = [
        synthetic_view_graph(
            3 + index % 6,
            family="CapNInterdigitalTee",
            active_count=6.0,
            active_length_um=20.0,
            active_width_um=2.0,
            active_gap_um=1.0,
            geometry_signal=-1.2 + 0.16 * index,
            target_kind="geometry",
        )
        for index in range(18)
    ]
    groups = [f"geometry-band-{index // 3}" for index in range(len(support))]
    gate = make_gate(
        control_foundation,
        cross_validation_folds=3,
        minimum_specialist_support=8,
        specialist_relative_margin=0.0,
        random_seed=43,
    ).fit(support, groups=groups)

    assert gate.choice == "specialist"
    assert gate.evidence_.fold_count == 3
    assert gate.evidence_.specialist_cv_log_mae < gate.evidence_.transfer_cv_log_mae
    assert gate.evidence_.specialist_paired_improvement > gate.evidence_.specialist_improvement_standard_error
    assert gate.specialist_ is not None and gate.adapter_ is None
    assert_physical_and_permutation_equivariant(gate, support[-1])


def test_enough_support_can_retain_transfer_when_specialist_evidence_is_insufficient(control_foundation):
    support = [
        synthetic_view_graph(
            3 + index % 7,
            family="CapNInterdigitalTee",
            active_count=3.0 + index % 8,
            active_length_um=12.0 + index,
            active_width_um=1.0 + 0.1 * (index % 6),
            active_gap_um=0.5 + 0.08 * (index % 5),
            geometry_signal=0.3 + 0.05 * index,
            target_kind="control",
        )
        for index in range(18)
    ]
    groups = [f"profile-{index // 3}" for index in range(len(support))]
    gate = EvidenceGatedTopoCap(
        control_foundation,
        build_topology_control_view,
        build_active_geometry_view,
        adapter_config=EBRAConfig(prior_precision=10.0),
        specialist_config=TopoCapConfig(random_feature_dimensions=0, ridge_alpha=1e8, random_seed=47),
        gate_config=EvidenceGateConfig(
            cross_validation_folds=3,
            minimum_specialist_support=8,
            specialist_relative_margin=0.25,
            random_seed=47,
        ),
    ).fit(support, groups=groups)

    assert gate.choice == "transfer"
    assert gate.evidence_.fold_count == 3
    assert gate.evidence_.transfer_cv_log_mae <= gate.evidence_.specialist_cv_log_mae / 0.75
    assert gate.adapter_ is not None and gate.specialist_ is None
    assert_physical_and_permutation_equivariant(gate, support[-1])


def test_grouped_cv_keeps_each_group_wholly_inside_one_validation_fold():
    groups = np.repeat(["a", "b", "c", "d", "e"], [3, 2, 4, 3, 2]).tolist()
    folds = _cross_validation_folds(len(groups), groups=groups, fold_count=3, seed=13)

    assert sorted(np.concatenate(folds).tolist()) == list(range(len(groups)))
    assert not any(set(left) & set(right) for position, left in enumerate(folds) for right in folds[position + 1 :])
    for group in set(groups):
        indices = set(np.flatnonzero(np.asarray(groups) == group).tolist())
        containing_folds = [fold for fold in folds if indices & set(fold.tolist())]
        assert len(containing_folds) == 1
        assert indices.issubset(set(containing_folds[0].tolist()))

    with pytest.raises(ValueError, match="at least two distinct"):
        _cross_validation_folds(4, groups=["same"] * 4, fold_count=3, seed=13)


def test_gate_retains_transfer_without_enough_independent_domains(control_foundation):
    support = [synthetic_view_graph(3 + index % 3) for index in range(12)]
    gate = make_gate(
        control_foundation,
        minimum_specialist_support=8,
        minimum_group_count=3,
    ).fit(support, groups=["domain-a"] * 6 + ["domain-b"] * 6)

    assert gate.choice == "transfer"
    assert gate.evidence_.fold_count == 0
    assert "Too few independent support domains" in gate.evidence_.reason


def test_gate_rejects_leaky_or_invalid_support_inputs(control_foundation):
    support = [synthetic_view_graph(3 + index % 3) for index in range(8)]
    gate = make_gate(control_foundation, minimum_specialist_support=8)

    with pytest.raises(ValueError, match="one label per support"):
        gate.fit(support, groups=["short"])
    gate.fit(support, groups=["one-profile"] * len(support))
    assert gate.choice == "transfer"
    assert "Too few independent support domains" in gate.evidence_.reason
    with pytest.raises(ValueError, match="capacitance target"):
        gate.fit([support[0].without_target()])


def test_gate_must_be_fitted_before_choice_or_prediction(control_foundation):
    gate = make_gate(control_foundation)
    with pytest.raises(RuntimeError, match="before reading its choice"):
        _ = gate.choice
    with pytest.raises(RuntimeError, match="before prediction"):
        gate.predict(synthetic_view_graph(3))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cross_validation_folds": 1},
        {"minimum_specialist_support": 1},
        {"minimum_group_count": 1},
        {"specialist_relative_margin": -0.1},
        {"specialist_relative_margin": 1.0},
        {"specialist_standard_error_margin": -0.1},
    ),
)
def test_evidence_gate_configuration_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        EvidenceGateConfig(**kwargs)
