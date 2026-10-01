from __future__ import annotations

import json

import pytest

from squadds.layouts.embedding_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    generate_controlled_planar_suite,
    radial_terminals,
    write_standard_planar_gds,
)
from squadds.layouts.geometry_v2 import read_layer_geometry


def test_arbitrary_terminal_gds_uses_additive_port_layers(tmp_path):
    pytest.importorskip("klayout.db")
    path = tmp_path / "six.gds"
    sidecar = write_standard_planar_gds(path, radial_terminals(6))
    geometry = read_layer_geometry(path)

    assert sidecar["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert sidecar["terminal_count"] == 6
    assert set(geometry) == {(1, 0), (1, 10), *( (2 + index, 0) for index in range(6) )}
    assert len(json.loads(path.with_suffix(".layers.json").read_text())["layers"]) == 8


def test_standard_ground_has_one_hole_and_every_port_bridges_moat(tmp_path):
    pytest.importorskip("klayout.db")
    path = tmp_path / "four.gds"
    write_standard_planar_gds(path, radial_terminals(4), clearance_um=5.0)
    geometry = read_layer_geometry(path)
    ground = geometry[(1, 0)]
    conductor = geometry[(1, 10)]

    polygons = list(getattr(ground, "geoms", [ground]))
    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1
    for index in range(4):
        port = geometry[(2 + index, 0)]
        assert port.distance(conductor) == pytest.approx(0.0, abs=1e-9)
        assert port.distance(ground) == pytest.approx(0.0, abs=1e-9)


def test_controlled_suite_is_complete_and_reproducible(tmp_path):
    pytest.importorskip("klayout.db")
    first = generate_controlled_planar_suite(tmp_path)
    hashes = {row["fixture_name"]: row["gds_sha256"] for row in first}
    second = generate_controlled_planar_suite(tmp_path)

    assert len(first) == 40
    assert {row["perturbation_group"] for row in first} == {
        "nuisance",
        "uniform_scale",
        "terminal_gap_um",
        "facing_length_um",
        "ground_clearance_um",
        "terminal_count",
    }
    assert hashes == {row["fixture_name"]: row["gds_sha256"] for row in second}

