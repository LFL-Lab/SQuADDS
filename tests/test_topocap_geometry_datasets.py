"""Synthetic GDS, sidecar, unit, catalogue, and cache integration tests."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from _topocap_helpers import generalized_row, physical_matrix, write_generalized_test_gds

from squadds.ml.topocap.datasets import (
    CACHE_JSONL_NAME,
    CacheVerificationError,
    CapacitanceUnitError,
    iter_cached_graphs,
    pair_generalized_records,
    parse_signed_maxwell_matrix_ff,
    verify_graph_cache,
    write_graph_cache,
)
from squadds.ml.topocap.geometry_graph import (
    EDGE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_geometry_graph_record,
    graph_record_arrays,
    to_capacitance_graph,
)
from squadds.ml.topocap.net_extraction import (
    NetSidecarError,
    build_generalized_ncap_sidecar,
    extract_sidecar_geometry,
    inventory_summary,
    read_gds_inventory,
    validate_net_sidecar,
)


@pytest.fixture
def generalized_gds(tmp_path):
    pytest.importorskip("gdstk")
    pytest.importorskip("shapely")
    return write_generalized_test_gds(tmp_path / "cap_0001.gds")


def test_explicit_unit_parser_normalizes_f_pf_and_ff():
    expected = physical_matrix(3, scale=2.5)
    for unit, divisor in (("fF", 1.0), ("pF", 1e3), ("F", 1e15)):
        row = generalized_row(expected / divisor)
        row["sim_results"]["units"] = unit
        np.testing.assert_allclose(parse_signed_maxwell_matrix_ff(row), expected)


@pytest.mark.parametrize("unit", (None, "nF", "farads", ""))
def test_unit_parser_rejects_missing_or_ambiguous_units(unit):
    row = generalized_row()
    if unit is None:
        del row["sim_results"]["units"]
    else:
        row["sim_results"]["units"] = unit
    with pytest.raises(CapacitanceUnitError):
        parse_signed_maxwell_matrix_ff(row)


def test_unit_parser_rejects_conflicting_tags():
    row = generalized_row()
    row["capacitance_unit"] = "pF"
    with pytest.raises(CapacitanceUnitError, match="Conflicting"):
        parse_signed_maxwell_matrix_ff(row)


def test_sidecar_maps_markers_to_nets_and_locks_the_gds_hash(generalized_gds):
    inventory = read_gds_inventory(generalized_gds)
    summary = inventory_summary(inventory)
    sidecar = build_generalized_ncap_sidecar(generalized_gds, inventory=inventory)
    extracted = extract_sidecar_geometry(generalized_gds, sidecar, inventory=inventory)

    assert summary["top_cell"] == "TOP"
    assert {(item["layer"], item["datatype"]) for item in summary["layers"]} == {
        (1, 0),
        (1, 10),
        (2, 0),
        (3, 0),
    }
    assert sidecar["matrix"]["node_order"] == ["net_000", "net_001", "net_002"]
    assert [net["is_reference"] for net in extracted["nets"]] == [True, False, False]
    assert [len(net["port_geometries"]) for net in extracted["nets"]] == [0, 1, 1]

    tampered = copy.deepcopy(sidecar)
    tampered["gds"]["sha256"] = "0" * 64
    tampered.pop("sidecar_sha256")
    with pytest.raises(NetSidecarError, match="hash"):
        validate_net_sidecar(tampered, inventory=inventory)


def test_geometry_graph_is_deterministic_finite_and_target_blind(generalized_gds):
    inventory = read_gds_inventory(generalized_gds)
    sidecar = build_generalized_ncap_sidecar(generalized_gds, inventory=inventory)
    kwargs = {
        "parameter_names": ("finger_count", "finger_gap"),
        "parameter_values": (5.0, 1.0),
        "parameter_features": ((1.0, 0.0), (0.0, 1.0)),
        "metadata": {"dataset_family": "synthetic-generalized"},
        "inventory": inventory,
        "boundary_sample_count": 16,
    }
    first = build_geometry_graph_record(generalized_gds, sidecar, **kwargs)
    second = build_geometry_graph_record(generalized_gds, sidecar, **kwargs)
    arrays = graph_record_arrays(first)
    graph = to_capacitance_graph(first)

    assert first["record_sha256"] == second["record_sha256"]
    assert first["capacitance_matrix_ff"] is None
    assert arrays["node_features"].shape == (3, len(NODE_FEATURE_NAMES))
    assert arrays["edge_features"].shape == (3, len(EDGE_FEATURE_NAMES))
    assert arrays["global_features"].shape == (len(GLOBAL_FEATURE_NAMES),)
    assert all(np.isfinite(arrays[name]).all() for name in ("node_features", "edge_features", "global_features"))
    assert graph.node_count == 3
    assert not graph.has_target
    assert graph.metadata["reference_crop"]["local_reference_area_um2"] > 0.0

    with pytest.raises(ValueError, match="at least 16"):
        build_geometry_graph_record(generalized_gds, sidecar, boundary_sample_count=15, inventory=inventory)


def test_small_catalogue_cache_round_trip_resume_and_corruption_detection(generalized_gds, tmp_path):
    exp6 = tmp_path / "exp6"
    exp6.mkdir()
    cache_gds = exp6 / generalized_gds.name
    cache_gds.write_bytes(generalized_gds.read_bytes())
    manifest = pair_generalized_records(
        [generalized_row()],
        exp6_gds_dir=exp6,
        exp7_gds_dir=tmp_path / "exp7",
        q3d_gds_root=tmp_path / "q3d",
        expected_count=1,
    )
    manifest.assert_complete()
    cache_dir = tmp_path / "cache"

    summary = write_graph_cache(manifest.entries, cache_dir, boundary_sample_count=16)
    verification = verify_graph_cache(manifest.entries, cache_dir, boundary_sample_count=16)
    resumed = write_graph_cache(
        manifest.entries,
        cache_dir,
        resume=True,
        boundary_sample_count=16,
    )
    graphs = list(iter_cached_graphs(cache_dir / CACHE_JSONL_NAME))
    raw_record = json.loads((cache_dir / CACHE_JSONL_NAME).read_text().strip())

    assert summary.written_count == 1
    assert verification.record_count == 1
    assert resumed.existing_count == 1
    assert resumed.written_count == 0
    assert len(graphs) == 1 and graphs[0].has_target
    assert raw_record["graph_record"]["capacitance_matrix_ff"] is None
    assert raw_record["target"]["unit"] == "fF"
    np.testing.assert_allclose(graphs[0].capacitance_matrix, physical_matrix(3, scale=2.0))

    raw_record["target"]["unit"] = "pF"
    (cache_dir / CACHE_JSONL_NAME).write_text(json.dumps(raw_record) + "\n")
    with pytest.raises(CacheVerificationError, match="hash|unit|manifest"):
        verify_graph_cache(manifest.entries, cache_dir, boundary_sample_count=16)
