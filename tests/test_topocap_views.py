"""Canonical control adapters and arbitrary-size TopoCap feature views."""

from __future__ import annotations

import numpy as np
import pytest
from _topocap_helpers import synthetic_view_graph

from squadds.ml.topocap.datasets import PARAMETER_DIMENSIONS, PARAMETER_FEATURE_NAMES, PARAMETER_ROLES
from squadds.ml.topocap.model import EquivariantFeatureBuilder
from squadds.ml.topocap.views import (
    ACTIVE_EDGE_FEATURES,
    ACTIVE_GLOBAL_FEATURES,
    ACTIVE_NODE_FEATURES,
    CAPN_INTERDIGITAL_TEE_CONTROLS,
    GENERALIZED_NCAP_CONTROLS,
    CanonicalControl,
    ControlSchema,
    build_active_geometry_view,
    build_topology_control_view,
    ncap_control_schema,
)


def test_built_in_ncap_adapters_produce_identical_canonical_tokens():
    generalized = synthetic_view_graph(3, family="GeneralizedCapNInterdigital")
    legacy = synthetic_view_graph(3, family="CapNInterdigitalTee")

    generalized_names, generalized_values, generalized_features = GENERALIZED_NCAP_CONTROLS.tokenize(generalized)
    legacy_names, legacy_values, legacy_features = CAPN_INTERDIGITAL_TEE_CONTROLS.tokenize(legacy)

    assert (
        generalized_names
        == legacy_names
        == (
            "active_count",
            "active_length_um",
            "active_width_um",
            "active_gap_um",
        )
    )
    np.testing.assert_allclose(generalized_values, [6.0, 20.0, 2.0, 1.0])
    np.testing.assert_array_equal(generalized_values, legacy_values)
    np.testing.assert_array_equal(generalized_features, legacy_features)
    assert generalized_features.shape == (4, len(PARAMETER_FEATURE_NAMES))

    count_row = generalized_features[0]
    assert count_row[PARAMETER_DIMENSIONS.index("count")] == 1.0
    role_offset = len(PARAMETER_DIMENSIONS) + 1
    gated_offset = role_offset + len(PARAMETER_ROLES)
    for role in ("count", "active_coupling_region"):
        role_index = PARAMETER_ROLES.index(role)
        assert count_row[role_offset + role_index] == 1.0
        assert count_row[gated_offset + role_index] == pytest.approx(6.0)


def test_custom_control_aggregations_are_explicit_and_validated():
    values = {"left": 2.0, "right": 6.0}
    assert CanonicalControl("mean", ("left", "right"), "length_um", ("length",), "mean").extract(values) == 4.0
    assert CanonicalControl("minimum", ("left", "right"), "length_um", ("length",), "minimum").extract(values) == 2.0
    assert CanonicalControl("maximum", ("left", "right"), "length_um", ("length",), "maximum").extract(values) == 6.0

    with pytest.raises(ValueError, match="exactly one"):
        CanonicalControl("bad", ("left", "right"), "length_um", ("length",))
    with pytest.raises(ValueError, match="Unknown parameter dimension"):
        CanonicalControl("bad", ("left",), "voltage", ("length",))
    with pytest.raises(ValueError, match="physical roles"):
        CanonicalControl("bad", ("left",), "length_um", ("native_name",))
    with pytest.raises(ValueError, match="unique"):
        ControlSchema(
            "duplicates",
            (
                CanonicalControl("same", ("left",), "length_um", ("length",)),
                CanonicalControl("same", ("right",), "length_um", ("length",)),
            ),
        )


@pytest.mark.parametrize("node_count", range(2, 17))
def test_views_preserve_arbitrary_topology_target_and_expected_widths(node_count):
    graph = synthetic_view_graph(node_count, geometry_signal=0.5 + node_count / 20.0)
    control = build_topology_control_view(graph)
    active = build_active_geometry_view(graph)
    geometry_only = build_active_geometry_view(graph, include_controls=False)

    for view in (control, active, geometry_only):
        assert view.node_count == node_count
        assert view.edge_count == node_count * (node_count - 1) // 2
        np.testing.assert_array_equal(view.edge_index, graph.edge_index)
        np.testing.assert_array_equal(view.capacitance_matrix, graph.capacitance_matrix)
        assert view.net_ids == graph.net_ids

    assert control.node_feature_dim == 1
    assert control.edge_feature_dim == 1
    assert control.global_feature_dim == 2
    assert control.parameter_names == (
        "active_count",
        "active_length_um",
        "active_width_um",
        "active_gap_um",
    )
    assert active.node_feature_dim == len(ACTIVE_NODE_FEATURES)
    assert active.edge_feature_dim == len(ACTIVE_EDGE_FEATURES)
    assert active.global_feature_dim == len(ACTIVE_GLOBAL_FEATURES)
    assert geometry_only.parameter_names == ()
    assert geometry_only.parameter_values.shape == (0,)
    assert geometry_only.parameter_features.shape == (0, 0)


def test_tool_specific_names_never_enter_the_numerical_model_view():
    generalized = build_topology_control_view(synthetic_view_graph(5, family="GeneralizedCapNInterdigital"))
    legacy = build_topology_control_view(synthetic_view_graph(5, family="CapNInterdigitalTee"))
    signature = EquivariantFeatureBuilder.infer_signature([generalized, legacy])
    generalized_rows = EquivariantFeatureBuilder.build(generalized, signature)
    legacy_rows = EquivariantFeatureBuilder.build(legacy, signature)

    assert generalized.parameter_names == legacy.parameter_names
    assert not any("finger" in name or "cap_" in name for name in generalized.parameter_names)
    assert generalized.parameter_features.dtype.kind == "f"
    assert legacy.parameter_features.dtype.kind == "f"
    for generalized_array, legacy_array in zip(generalized_rows, legacy_rows):
        np.testing.assert_array_equal(generalized_array, legacy_array)
        assert generalized_array.dtype.kind == "f"
    assert "finger_count" in generalized.metadata["split_parameter_values"]
    assert "cap_width" in legacy.metadata["split_parameter_values"]


@pytest.mark.parametrize("node_count", (2, 3, 7, 16))
@pytest.mark.parametrize("builder", (build_topology_control_view, build_active_geometry_view))
def test_view_construction_commutes_with_node_permutation(node_count, builder):
    graph = synthetic_view_graph(node_count, geometry_signal=1.4)
    order = np.random.default_rng(100 + node_count).permutation(node_count)
    expected = builder(graph).reorder_nodes(order)
    actual = builder(graph.reorder_nodes(order))

    np.testing.assert_array_equal(actual.node_features, expected.node_features)
    np.testing.assert_array_equal(actual.edge_features, expected.edge_features)
    np.testing.assert_array_equal(actual.global_features, expected.global_features)
    np.testing.assert_array_equal(actual.parameter_values, expected.parameter_values)
    np.testing.assert_array_equal(actual.capacitance_matrix, expected.capacitance_matrix)


def test_missing_or_unknown_control_mappings_fail_loudly():
    unknown = synthetic_view_graph(3, family="UnregisteredLayoutTool")
    with pytest.raises(KeyError, match="No built-in NCap control schema"):
        ncap_control_schema(unknown)

    missing_metadata = synthetic_view_graph(3)
    metadata = dict(missing_metadata.metadata)
    metadata.pop("split_parameter_values")
    missing_metadata = type(missing_metadata)(
        node_features=missing_metadata.node_features,
        edge_index=missing_metadata.edge_index,
        edge_features=missing_metadata.edge_features,
        global_features=missing_metadata.global_features,
        net_ids=missing_metadata.net_ids,
        capacitance_matrix=missing_metadata.capacitance_matrix,
        metadata=metadata,
    )
    with pytest.raises(KeyError, match="split_parameter_values"):
        build_topology_control_view(missing_metadata)

    incomplete = synthetic_view_graph(3)
    metadata = dict(incomplete.metadata)
    metadata["split_parameter_values"] = {"finger_count": 6.0}
    incomplete = type(incomplete)(
        node_features=incomplete.node_features,
        edge_index=incomplete.edge_index,
        edge_features=incomplete.edge_features,
        global_features=incomplete.global_features,
        net_ids=incomplete.net_ids,
        capacitance_matrix=incomplete.capacitance_matrix,
        metadata=metadata,
    )
    with pytest.raises(KeyError, match="Missing source fields"):
        build_topology_control_view(incomplete)


def test_unknown_family_can_use_an_explicit_canonical_adapter():
    graph = synthetic_view_graph(4, family="UnregisteredLayoutTool")
    schema = ControlSchema(
        "portable-test-schema",
        (
            CanonicalControl("active_count", ("native_count",), "count", ("count",)),
            CanonicalControl("active_length_um", ("native_length",), "length_um", ("length",)),
            CanonicalControl("active_width_um", ("native_width",), "length_um", ("width",)),
            CanonicalControl("active_gap_um", ("native_gap",), "length_um", ("gap",)),
        ),
    )
    view = build_topology_control_view(graph, schema=schema)

    assert view.parameter_names == tuple(control.name for control in schema.controls)
    np.testing.assert_allclose(view.parameter_values, [6.0, 20.0, 2.0, 1.0])
    assert view.metadata["topocap_view"]["control_schema"] == "portable-test-schema"
