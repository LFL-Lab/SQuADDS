from __future__ import annotations

import json

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import box

from squadds.layouts.embedding_benchmark import radial_terminals, rectangular_pair, write_standard_planar_gds
from squadds.layouts.geometry_v4 import (
    V4_EDGE_DIMENSIONS,
    V4_GLOBAL_DIMENSIONS,
    V4_HYBRID_PAIR_DIMENSIONS,
    V4_LAYOUT_VIEW_DIMENSIONS,
    V4_NODE_DIMENSIONS,
    V4_PAIR_VIEW_DIMENSIONS,
    encode_v4_graph,
    encode_v4_pair,
    layout_view_v4,
    pair_view_v4,
    planar_terminal_graph_v4_schema,
)


def _write(tmp_path, name, terminals, **kwargs):
    path = tmp_path / f"{name}.gds"
    write_standard_planar_gds(path, terminals, **kwargs)
    return path


def test_v4_has_no_terminal_count_cap(tmp_path):
    pytest.importorskip("klayout.db")
    path = _write(tmp_path, "sixteen", radial_terminals(16, radius_um=90.0, pad_size_um=10.0))
    graph = encode_v4_graph(path, layer_contract=path.with_suffix(".layers.json"))

    assert graph.node_features.shape == (16, V4_NODE_DIMENSIONS)
    assert graph.edge_features.shape == (16, 16, V4_EDGE_DIMENSIONS)
    assert graph.global_features.shape == (V4_GLOBAL_DIMENSIONS,)
    assert graph.queryable_mask.all()
    assert graph.metadata["terminal_count"] == 16
    assert np.array_equal(graph.edge_features, graph.edge_features.swapaxes(0, 1))
    assert layout_view_v4(graph).shape == (V4_LAYOUT_VIEW_DIMENSIONS,)
    assert pair_view_v4(graph, 0, 15).shape == (V4_PAIR_VIEW_DIMENSIONS,)


def test_v4_is_deterministic_and_layout_view_is_permutation_invariant(tmp_path):
    pytest.importorskip("klayout.db")
    path = _write(tmp_path, "four", radial_terminals(4))
    contract_path = path.with_suffix(".layers.json")
    original = encode_v4_graph(path, layer_contract=contract_path)
    repeated = encode_v4_graph(path, layer_contract=contract_path)
    assert np.array_equal(layout_view_v4(original), layout_view_v4(repeated))

    contract = json.loads(contract_path.read_text())
    for layer in contract["layers"]:
        if layer["role"] == "port":
            layer["terminal_index"] = 3 - layer["terminal_index"]
            layer["terminal_id"] = f"r{layer['terminal_index']}"
            layer["net_id"] = layer["terminal_id"]
    relabelled = encode_v4_graph(path, layer_contract=contract)
    assert np.allclose(layout_view_v4(original), layout_view_v4(relabelled), rtol=1e-7, atol=1e-7)


def test_pair_view_is_reciprocal_and_preserves_other_terminal_context(tmp_path):
    pytest.importorskip("klayout.db")
    base = radial_terminals(3, radius_um=50.0)
    near_path = _write(tmp_path, "near", base)
    moved = [base[0], base[1], affinity.translate(base[2], xoff=80.0)]
    far_path = _write(tmp_path, "far", moved)
    near = encode_v4_graph(near_path, layer_contract=near_path.with_suffix(".layers.json"))
    far = encode_v4_graph(far_path, layer_contract=far_path.with_suffix(".layers.json"))

    assert np.array_equal(pair_view_v4(near, 0, 1), pair_view_v4(near, 1, 0))
    assert not np.array_equal(pair_view_v4(near, 0, 1), pair_view_v4(far, 0, 1))
    hybrid_near = encode_v4_pair(near_path, 0, 1, layer_contract=near_path.with_suffix(".layers.json"), graph=near)
    hybrid_reverse = encode_v4_pair(near_path, 1, 0, layer_contract=near_path.with_suffix(".layers.json"), graph=near)
    hybrid_far = encode_v4_pair(far_path, 0, 1, layer_contract=far_path.with_suffix(".layers.json"), graph=far)
    assert hybrid_near.shape == (V4_HYBRID_PAIR_DIMENSIONS,)
    assert np.array_equal(hybrid_near, hybrid_reverse)
    assert not np.array_equal(hybrid_near[240:], hybrid_far[240:])


def test_v4_sidecar_disambiguates_a_nested_terminal_marker(tmp_path):
    pytest.importorskip("klayout.db")
    inner = box(-5.0, -5.0, 5.0, 5.0)
    outer = box(-25.0, -25.0, 25.0, 25.0).difference(box(-15.0, -15.0, 15.0, 15.0))
    path = _write(tmp_path, "nested", [inner, outer], clearance_um=6.0)

    graph = encode_v4_graph(path, layer_contract=path.with_suffix(".layers.json"))

    assert graph.metadata["terminal_count"] == 2
    assert graph.metadata["unmarked_conductor_count"] == 0


@pytest.mark.parametrize(
    ("name", "transform", "rtol"),
    [
        ("translated", lambda shape: affinity.translate(shape, xoff=4000.0, yoff=-2500.0), 1e-6),
        ("rotated90", lambda shape: affinity.rotate(shape, 90.0, origin=(0.0, 0.0)), 8e-4),
        ("rotated37", lambda shape: affinity.rotate(shape, 37.0, origin=(0.0, 0.0)), 4e-3),
        ("reflected", lambda shape: affinity.scale(shape, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)), 8e-4),
    ],
)
def test_v4_layout_view_is_rigid_motion_invariant(tmp_path, name, transform, rtol):
    pytest.importorskip("klayout.db")
    base = rectangular_pair()
    first_path = _write(tmp_path, "base", base)
    other_path = _write(tmp_path, name, [transform(shape) for shape in base])
    first = layout_view_v4(encode_v4_graph(first_path, layer_contract=first_path.with_suffix(".layers.json")))
    other = layout_view_v4(encode_v4_graph(other_path, layer_contract=other_path.with_suffix(".layers.json")))
    relative = np.linalg.norm(first - other) / max(np.linalg.norm(first), 1e-12)
    assert relative <= rtol


def test_outer_ground_crop_is_excluded_but_clearance_is_physical(tmp_path):
    pytest.importorskip("klayout.db")
    terminals = rectangular_pair()
    small = _write(tmp_path, "small", terminals, domain_padding_um=50.0)
    large = _write(tmp_path, "large", terminals, domain_padding_um=500.0)
    wider = _write(tmp_path, "wider", terminals, clearance_um=10.0)
    a = layout_view_v4(encode_v4_graph(small, layer_contract=small.with_suffix(".layers.json")))
    b = layout_view_v4(encode_v4_graph(large, layer_contract=large.with_suffix(".layers.json")))
    c = layout_view_v4(encode_v4_graph(wider, layer_contract=wider.with_suffix(".layers.json")))

    assert np.allclose(a, b, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(a, c)


def test_schema_is_catalogue_free_and_rejects_multiplane_contract(tmp_path):
    pytest.importorskip("klayout.db")
    schema = planar_terminal_graph_v4_schema()
    assert schema["fitted_on_catalogue"] is False
    assert schema["simulation_results_used"] is False
    assert schema["canonical_representation"]["terminal_count_cap"] is None

    path = _write(tmp_path, "pair", rectangular_pair())
    contract = json.loads(path.with_suffix(".layers.json").read_text())
    contract["planes"] = [{"plane_id": "a", "z_um": 0.0}, {"plane_id": "b", "z_um": 10.0}]
    with pytest.raises(ValueError, match="exactly one physical conductor plane"):
        encode_v4_graph(path, layer_contract=contract)


def test_unmarked_conductor_is_preserved_as_queryable_context(tmp_path):
    pytest.importorskip("klayout.db")
    path = _write(tmp_path, "pair_unmarked", rectangular_pair())
    contract = json.loads(path.with_suffix(".layers.json").read_text())
    for layer in contract["layers"]:
        if layer.get("terminal_index") == 1:
            layer["role"] = "ignored"
    graph = encode_v4_graph(path, layer_contract=contract)

    assert graph.metadata["terminal_count"] == 2
    assert graph.metadata["unmarked_conductor_count"] == 1
    assert graph.node_roles == ("terminal", "unmarked")
    assert graph.queryable_mask.all()
    assert pair_view_v4(graph, 0, 1).shape == (V4_PAIR_VIEW_DIMENSIONS,)
