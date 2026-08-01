"""Contracts for arbitrary-size TopoCap graph records and physical targets."""

from __future__ import annotations

import numpy as np
import pytest
from _topocap_helpers import physical_matrix, synthetic_graph

from squadds.ml.topocap.schema import (
    CapacitanceGraph,
    canonical_edge_index,
    inverse_permutation,
)
from squadds.ml.topocap.targets import (
    components_to_maxwell,
    logs_to_maxwell,
    maxwell_diagnostics,
    maxwell_to_components,
)


@pytest.mark.parametrize("node_count", range(2, 17))
def test_maxwell_factorization_is_physical_and_exact_for_arbitrary_node_counts(node_count):
    matrix = physical_matrix(node_count, scale=0.7 + node_count / 10.0)
    components = maxwell_to_components(matrix)
    diagnostics = maxwell_diagnostics(matrix)

    assert components.edge_index.shape == (2, node_count * (node_count - 1) // 2)
    assert np.all(components.shunts > 0.0)
    assert np.all(components.mutuals > 0.0)
    assert diagnostics.is_physical
    np.testing.assert_allclose(components.to_matrix(), matrix, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        logs_to_maxwell(components.log_shunts, components.log_mutuals),
        matrix,
        rtol=1e-13,
        atol=1e-13,
    )


def test_component_reconstruction_guarantees_signed_maxwell_properties():
    rng = np.random.default_rng(91)
    for node_count in range(2, 17):
        edge_count = node_count * (node_count - 1) // 2
        matrix = components_to_maxwell(
            np.exp(rng.normal(size=node_count)),
            np.exp(rng.normal(size=edge_count)),
        )

        np.testing.assert_allclose(matrix, matrix.T)
        assert np.all(np.diag(matrix) > 0.0)
        assert np.all(matrix[np.triu_indices(node_count, 1)] < 0.0)
        assert np.min(matrix.sum(axis=1)) > 0.0
        assert np.min(np.linalg.eigvalsh(matrix)) > 0.0


@pytest.mark.parametrize("node_count", (2, 3, 7, 16))
def test_graph_reordering_preserves_features_targets_and_bookkeeping(node_count):
    graph = synthetic_graph(node_count, geometry_scale=1.3)
    order = np.random.default_rng(node_count).permutation(node_count)
    reordered = graph.reorder_nodes(order)
    inverse = inverse_permutation(order)

    assert reordered.edge_index.tolist() == canonical_edge_index(node_count).tolist()
    assert reordered.net_ids == tuple(graph.net_ids[index] for index in order)
    np.testing.assert_allclose(reordered.node_features, graph.node_features[order])
    np.testing.assert_allclose(
        reordered.capacitance_matrix,
        graph.capacitance_matrix[np.ix_(order, order)],
    )
    np.testing.assert_allclose(reordered.reorder_nodes(inverse).node_features, graph.node_features)
    np.testing.assert_allclose(
        reordered.reorder_nodes(inverse).capacitance_matrix,
        graph.capacitance_matrix,
    )
    assert not graph.node_features.flags.writeable
    assert not graph.edge_features.flags.writeable
    assert not graph.capacitance_matrix.flags.writeable


def test_graph_supports_no_parameter_tokens_and_target_blind_copy():
    graph = CapacitanceGraph(
        node_features=np.ones((3, 2)),
        edge_index=canonical_edge_index(3),
        edge_features=np.empty((3, 0)),
        parameter_features=np.empty((0, 0)),
        capacitance_matrix=physical_matrix(3),
    )

    assert graph.parameter_features.shape == (0, 0)
    assert graph.without_target().capacitance_matrix is None
    assert graph.without_target().net_ids == graph.net_ids


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.asarray([[2.0, -1.0], [-0.5, 2.0]]), "symmetric"),
        (np.asarray([[2.0, 0.1], [0.1, 2.0]]), "off-diagonal"),
        (np.asarray([[1.0, -1.0], [-1.0, 1.0]]), "residual shunts"),
    ],
)
def test_invalid_maxwell_targets_are_rejected(matrix, message):
    with pytest.raises(ValueError, match=message):
        maxwell_to_components(matrix)


def test_graph_rejects_noncanonical_or_incomplete_edges():
    with pytest.raises(ValueError, match="canonical"):
        CapacitanceGraph(
            node_features=np.ones((3, 1)),
            edge_index=np.asarray([[0, 0], [1, 2]]),
            edge_features=np.ones((2, 1)),
        )


@pytest.mark.parametrize("bad_count", (True, 1, 2.5))
def test_canonical_edge_index_rejects_invalid_node_counts(bad_count):
    expected = TypeError if bad_count is True or bad_count == 2.5 else ValueError
    with pytest.raises(expected):
        canonical_edge_index(bad_count)
