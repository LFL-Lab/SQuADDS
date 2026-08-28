from __future__ import annotations

import pytest

from squadds.layouts.layer_semantics import LAYER_SEMANTICS, LAYER_SEMANTICS_SCHEMA_VERSION


def _metal():
    pytest.importorskip("klayout.db")
    pytest.importorskip("qiskit_metal")
    from scripts.generate_simulation_layout_gds import _enable_qiskit_metal_pandas_compatibility

    _enable_qiskit_metal_pandas_compatibility()
    from qiskit_metal import designs

    return designs


def test_transmon_cross_export_roundtrips_qgeometry_and_orders_ports(tmp_path):
    designs = _metal()
    from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross

    from squadds.layouts.geometry_v2 import read_layer_geometry
    from squadds.layouts.qmetal_gds import export_qgeometry_gds, transmon_cross_port_markers, validate_ported_gds

    design = designs.DesignPlanar()
    component = TransmonCross(
        design,
        "qubit",
        options={"orientation": "-90", "connection_pads": {"readout": {"connector_location": "90"}}},
    )
    markers = transmon_cross_port_markers(component, design)
    path = tmp_path / "transmon.gds"
    export_qgeometry_gds(design, path, markers=markers)

    assert [(marker.layer, marker.semantic) for marker in markers] == [
        (2, "cross_junction_port"),
        (3, "readout_claw_port"),
    ]
    assert validate_ported_gds(design, path, markers)["valid"] is True
    assert set(read_layer_geometry(path)) == {(1, 10), (1, 11), (2, 0), (3, 0)}

    repeat = tmp_path / "transmon-repeat.gds"
    export_qgeometry_gds(design, repeat, markers=markers)
    assert path.read_bytes() == repeat.read_bytes()


def test_capn_export_includes_every_metal_path_ground_and_two_ports(tmp_path):
    designs = _metal()
    from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

    from squadds.layouts.geometry_v2 import read_layer_geometry
    from squadds.layouts.qmetal_gds import (
        capn_interdigital_tee_port_markers,
        export_qgeometry_gds,
        validate_ported_gds,
    )

    design = designs.DesignPlanar()
    component = CapNInterdigitalTee(design, "cplr", options={"orientation": "-90", "finger_count": "5"})
    markers = capn_interdigital_tee_port_markers(component)
    path = tmp_path / "capn.gds"
    export_qgeometry_gds(design, path, markers=markers, include_ground_domain=True)

    report = validate_ported_gds(design, path, markers)
    assert report["valid"] is True
    assert report["signal_component_count"] == 2
    assert len(set(report["port_component_assignments"])) == 2
    assert set(read_layer_geometry(path)) == {(1, 0), (1, 10), (1, 11), (2, 0), (3, 0)}


def test_layer_semantics_declares_ordered_ports_for_both_retrofits():
    assert LAYER_SEMANTICS_SCHEMA_VERSION == "1.2.0"
    components = LAYER_SEMANTICS["components"]
    assert [entry["semantic"] for entry in components["TransmonCross"][-2:]] == [
        "cross_junction_port",
        "readout_claw_port",
    ]
    assert [entry["semantic"] for entry in components["CapNInterdigitalTee"][-2:]] == [
        "prime_top_port",
        "second_bottom_port",
    ]
