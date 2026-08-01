"""Permutation, physicality, serialization, and EBRA adaptation tests."""

from __future__ import annotations

import numpy as np
import pytest
from _topocap_helpers import rescale_target, synthetic_graph

from squadds.ml.topocap.adaptation import (
    EBRAAdapter,
    EBRAConfig,
    adapt_foundation,
    foundation_fingerprint,
)
from squadds.ml.topocap.model import TopoCapConfig, TopoCapFoundationModel
from squadds.ml.topocap.targets import maxwell_diagnostics, maxwell_to_components


@pytest.fixture(scope="module")
def training_graphs():
    return tuple(
        synthetic_graph(
            node_count,
            geometry_scale=0.75 + 0.08 * node_count,
            target_scale=0.9 + 0.03 * node_count,
        )
        for node_count in range(2, 17)
    )


@pytest.fixture(scope="module")
def foundation(training_graphs):
    return TopoCapFoundationModel(TopoCapConfig(random_feature_dimensions=0, ridge_alpha=0.2, random_seed=11)).fit(
        training_graphs
    )


def test_foundation_predicts_physical_variable_size_matrices(foundation, training_graphs):
    predictions = foundation.predict_many(training_graphs, confidence=0.8)

    assert len(predictions) == 15
    for graph, prediction in zip(training_graphs, predictions):
        assert prediction.matrix.shape == (graph.node_count, graph.node_count)
        assert prediction.matrix_variance.shape == prediction.matrix.shape
        assert maxwell_diagnostics(prediction.matrix).is_physical
        assert maxwell_diagnostics(prediction.mean_matrix).is_physical
        lower, upper = prediction.interval()
        assert np.all(lower <= prediction.matrix + 1e-12)
        assert np.all(prediction.matrix <= upper + 1e-12)


@pytest.mark.parametrize("node_count", (2, 3, 6, 11, 16))
def test_foundation_prediction_is_permutation_equivariant(foundation, node_count):
    graph = synthetic_graph(node_count, geometry_scale=1.19, target_scale=1.07)
    order = np.random.default_rng(400 + node_count).permutation(node_count)

    original = foundation.predict(graph)
    permuted = foundation.predict(graph.reorder_nodes(order))

    np.testing.assert_allclose(
        permuted.matrix,
        original.matrix[np.ix_(order, order)],
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        permuted.mean_matrix,
        original.mean_matrix[np.ix_(order, order)],
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        permuted.matrix_variance,
        original.matrix_variance[np.ix_(order, order)],
        rtol=2e-11,
        atol=2e-11,
    )


def test_foundation_archive_round_trip_is_prediction_exact(foundation, tmp_path):
    path = foundation.save(tmp_path / "foundation.npz")
    restored = TopoCapFoundationModel.load(path)
    graph = synthetic_graph(9, geometry_scale=1.31)

    assert foundation_fingerprint(restored) == foundation_fingerprint(foundation)
    np.testing.assert_array_equal(restored.predict(graph).matrix, foundation.predict(graph).matrix)
    np.testing.assert_array_equal(
        restored.predict(graph).matrix_variance,
        foundation.predict(graph).matrix_variance,
    )


def test_ebra_reduces_controlled_target_domain_residual_without_mutating_foundation(
    foundation,
):
    target_graphs = [
        rescale_target(
            synthetic_graph(node_count, geometry_scale=0.9 + 0.04 * node_count),
            shunt_factor=np.exp(0.55),
            mutual_factor=np.exp(-0.35),
        )
        for node_count in range(3, 12)
    ]
    fingerprint_before = foundation_fingerprint(foundation)
    adapter = adapt_foundation(
        foundation,
        target_graphs[:6],
        EBRAConfig(prior_precision=0.05),
    )

    def latent_error(predictor, graphs):
        errors = []
        for graph in graphs:
            truth = maxwell_to_components(graph.capacitance_matrix)
            prediction = predictor.predict_latent(graph)
            errors.extend(prediction.log_shunt_mean - truth.log_shunts)
            errors.extend(prediction.log_mutual_mean - truth.log_mutuals)
        return float(np.mean(np.square(errors)))

    assert latent_error(adapter, target_graphs) < latent_error(foundation, target_graphs)
    assert foundation_fingerprint(foundation) == fingerprint_before
    assert adapter.support_graph_count_ == 6
    assert maxwell_diagnostics(adapter.predict(target_graphs[-1]).matrix).is_physical


def test_adapter_archive_requires_the_exact_foundation(foundation, training_graphs, tmp_path):
    support = [rescale_target(graph, shunt_factor=1.2, mutual_factor=0.9) for graph in training_graphs[:3]]
    adapter = EBRAAdapter(foundation, EBRAConfig(prior_precision=1.0)).fit(support)
    path = adapter.save(tmp_path / "adapter.npz")

    restored = EBRAAdapter.load(path, foundation)
    probe = training_graphs[4]
    np.testing.assert_array_equal(restored.predict(probe).matrix, adapter.predict(probe).matrix)

    other_foundation = TopoCapFoundationModel(
        TopoCapConfig(random_feature_dimensions=0, ridge_alpha=0.3, random_seed=11)
    ).fit(training_graphs)
    with pytest.raises(RuntimeError, match="foundation model has changed|Foundation model has changed"):
        EBRAAdapter.load(path, other_foundation)


def test_unfitted_models_and_target_blind_support_fail_loudly(foundation):
    graph = synthetic_graph(3)
    with pytest.raises(RuntimeError, match="Fit the foundation"):
        TopoCapFoundationModel().predict(graph)
    with pytest.raises(ValueError, match="capacitance_matrix"):
        EBRAAdapter(foundation).fit([graph.without_target()])
