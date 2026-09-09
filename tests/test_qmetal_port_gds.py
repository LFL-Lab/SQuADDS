from __future__ import annotations

import numpy as np
import pytest

from squadds.layouts.layer_semantics import (
    LAYER_SEMANTICS,
    LAYER_SEMANTICS_SCHEMA_VERSION,
    PORT_COMPLETE_ROLE_PROFILE,
    PUBLISHED_ROLE_PROFILE,
    functional_layer_roles,
)


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

    from squadds.layouts.embeddings import rasterize_functional_shape
    from squadds.layouts.geometry_v2 import read_layer_geometry
    from squadds.layouts.qmetal_gds import (
        GROUND_DOMAIN_PADDING_UM,
        export_qgeometry_gds,
        minimum_ground_clearance_um,
        transmon_cross_port_markers,
        validate_ported_gds,
    )

    design = designs.DesignPlanar()
    component = TransmonCross(
        design,
        "qubit",
        options={"orientation": "-90", "connection_pads": {"readout": {"connector_location": "90"}}},
    )
    markers = transmon_cross_port_markers(component, design)
    path = tmp_path / "transmon.gds"
    clearance = minimum_ground_clearance_um(component)
    export_qgeometry_gds(
        design,
        path,
        markers=markers,
        include_ground_domain=True,
        minimum_ground_clearance_um=clearance,
    )

    assert [(marker.layer, marker.semantic) for marker in markers] == [
        (2, "cross_junction_port"),
        (3, "readout_claw_port"),
    ]
    report = validate_ported_gds(design, path, markers, minimum_ground_clearance_um=clearance)
    assert report["valid"] is True
    assert report["ground_hole_count"] == 1
    assert report["ground_padding_um"] == pytest.approx(GROUND_DOMAIN_PADDING_UM, rel=0.05)
    assert len(set(report["port_component_assignments"])) == 2
    assert report["port_conductor_distances_um"] == [0.0, 0.0]
    assert report["port_ground_distances_um"] == [0.0, 0.0]
    assert set(read_layer_geometry(path)) == {(1, 0), (1, 10), (2, 0), (3, 0)}

    published, _, published_metadata = rasterize_functional_shape(
        path,
        "TransmonCross",
        role_profile=PUBLISHED_ROLE_PROFILE,
    )
    port_complete, _, port_complete_metadata = rasterize_functional_shape(
        path,
        "TransmonCross",
        role_profile=PORT_COMPLETE_ROLE_PROFILE,
    )
    assert {entry["role"] for entry in published_metadata["functional_layers"]} == {"conductor"}
    assert {entry["role"] for entry in port_complete_metadata["functional_layers"]} == {
        "conductor",
        "port",
    }
    assert not np.array_equal(published, port_complete)

    repeat = tmp_path / "transmon-repeat.gds"
    export_qgeometry_gds(
        design,
        repeat,
        markers=markers,
        include_ground_domain=True,
        minimum_ground_clearance_um=clearance,
    )
    assert path.read_bytes() == repeat.read_bytes()


def test_capn_export_includes_every_metal_path_ground_and_two_ports(tmp_path):
    designs = _metal()
    from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

    from squadds.layouts.geometry_v2 import read_layer_geometry
    from squadds.layouts.qmetal_gds import (
        GROUND_DOMAIN_PADDING_UM,
        capn_interdigital_tee_port_markers,
        export_qgeometry_gds,
        minimum_ground_clearance_um,
        validate_ported_gds,
    )

    design = designs.DesignPlanar()
    component = CapNInterdigitalTee(design, "cplr", options={"orientation": "-90", "finger_count": "5"})
    markers = capn_interdigital_tee_port_markers(component)
    clearance = minimum_ground_clearance_um(component)
    path = tmp_path / "capn.gds"
    export_qgeometry_gds(
        design,
        path,
        markers=markers,
        include_ground_domain=True,
        minimum_ground_clearance_um=clearance,
    )

    report = validate_ported_gds(design, path, markers, minimum_ground_clearance_um=clearance)
    assert report["valid"] is True
    assert report["signal_component_count"] == 2
    assert len(set(report["port_component_assignments"])) == 2
    assert report["ground_hole_count"] == 1
    assert report["ground_padding_um"] == pytest.approx(GROUND_DOMAIN_PADDING_UM, rel=0.05)
    assert report["port_ground_distances_um"] == [0.0, 0.0]
    assert set(read_layer_geometry(path)) == {(1, 0), (1, 10), (2, 0), (3, 0)}


def test_layer_semantics_declares_ordered_ports_for_both_retrofits():
    from squadds.layouts.embeddings import static_embedding_schema

    assert LAYER_SEMANTICS_SCHEMA_VERSION == "1.3.0"
    components = LAYER_SEMANTICS["components"]
    assert [entry["semantic"] for entry in components["TransmonCross"][-2:]] == [
        "cross_junction_port",
        "readout_claw_port",
    ]
    assert [entry["semantic"] for entry in components["CapNInterdigitalTee"][-2:]] == [
        "prime_top_port",
        "second_bottom_port",
    ]
    assert functional_layer_roles("TransmonCross") == {
        (1, 10): "conductor",
        (2, 0): "port",
        (3, 0): "port",
    }
    schema = static_embedding_schema(
        0.0,
        1.0,
        np.zeros(10),
        np.ones(10),
        role_profile=PORT_COMPLETE_ROLE_PROFILE,
    )
    assert schema["model"] == "static-shape-v0-port-complete"
    assert schema["shape_rasterization"]["role_profile"] == PORT_COMPLETE_ROLE_PROFILE
