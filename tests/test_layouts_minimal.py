import subprocess
import sys

import pandas as pd
import pytest
from qiskit_metal import designs

import squadds.layouts as layouts
from squadds.layouts import (
    LAYER_SEMANTICS,
    LayoutClient,
    PortMarker,
    canonical_design_id,
    export_qgeometry_gds,
    functional_layer_roles,
    parse_gds_polygons,
    parse_gds_summary,
)


def test_import_stays_minimal():
    code = (
        "import sys, squadds, squadds.core.analysis, squadds.core.db\n"  # the modules `import squadds` loaded eagerly
        "before = set(sys.modules)\n"
        "import squadds.layouts\n"
        "print(','.join(sorted(m for m in set(sys.modules) - before if not m.startswith('squadds.layouts'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
    assert all(hasattr(layouts, name) for name in layouts.__all__)


def test_canonical_design_id_is_order_independent():
    first = canonical_design_id("GeneralizedCapNInterdigital", {"finger_count": "5", "width": "10um"})
    second = canonical_design_id("GeneralizedCapNInterdigital", {"width": "10um", "finger_count": "5"})
    assert first == second
    assert first.startswith("design:sha256:")


def test_layer_semantics_cover_port_complete_layers():
    assert LAYER_SEMANTICS
    roles = functional_layer_roles("GeneralizedCapNInterdigital")
    assert roles
    assert all(isinstance(key, tuple) and len(key) == 2 for key in roles)


def test_layout_client_finds_local_manifest_record(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    artifact = tmp_path / "raw" / "cap.gds"
    artifact.parent.mkdir()
    artifact.write_bytes(b"gds")
    pd.DataFrame(
        [
            {
                "layout_id": "layout:sha256:layout",
                "artifact_id": "sha256:8261e1a4f86abe76f7a1e8a66ae713e37dc9e651408b95b7db70ea062d5fc071",
                "gds_path": "raw/cap.gds",
                "component_name": "GeneralizedCapNInterdigital",
                "source_id": "exp6/cap_0000",
                "design_id": "design:sha256:design",
            }
        ]
    ).to_parquet(manifest, index=False)

    client = LayoutClient(manifest_path=manifest, artifact_root=tmp_path)
    reference = client.find(source_id="exp6/cap_0000")

    assert reference.gds_path == "raw/cap.gds"
    assert reference.component_name == "GeneralizedCapNInterdigital"
    assert client.download(reference) == artifact


def test_parse_gds_summary_extracts_layers_and_units(tmp_path):
    kdb = pytest.importorskip("klayout.db")
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    top.shapes(layout.layer(1, 10)).insert(kdb.Box(0, 0, 1000, 2000))
    gds_path = tmp_path / "layout.gds"
    layout.write(str(gds_path))

    summary = parse_gds_summary(gds_path)

    assert summary["top_cell"] == "TOP"
    assert summary["dbu_um"] == 0.001
    assert summary["polygon_count"] == 1
    assert summary["layers"][0]["area_um2"] == 2.0
    assert parse_gds_polygons(gds_path, layer=1, datatype=10)[0]["area_um2"] == 2.0


def test_export_qgeometry_gds_writes_ground_signal_and_ports(tmp_path):
    pytest.importorskip("klayout.db")
    from shapely.geometry import box

    from squadds.components import GeneralizedCapNInterdigital
    from squadds.layouts.qmetal_gds import read_layer_geometry

    design = designs.DesignPlanar()
    component = GeneralizedCapNInterdigital(design, "cap", options={})
    markers = []
    for layer, pin in ((2, "north_end"), (3, "south_end")):
        x, y = component.pins[pin]["middle"]
        nx, ny = component.pins[pin]["normal"]
        w = component.pins[pin]["width"]
        markers.append(
            PortMarker(pin, layer, box(x - w / 2, min(y, y + 0.002 * ny), x + w / 2, max(y, y + 0.002 * ny)), pin)
        )
    path = tmp_path / "cap.gds"
    export_qgeometry_gds(design, path, markers=markers, include_ground_domain=True, minimum_ground_clearance_um=6.0)

    geometry = read_layer_geometry(path)
    assert {(1, 0), (1, 10), (2, 0), (3, 0)} <= set(geometry)
    assert len(geometry[(1, 10)].geoms) == 2
    assert geometry[(1, 0)].geom_type == "Polygon" and len(geometry[(1, 0)].interiors) == 1
