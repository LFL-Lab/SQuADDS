"""squadds.layouts.dataset_export: the planar capacitance dataset GDS convention (CI-safe; dataset parity is an
integration test on local tl-squadds data)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

gdstk = pytest.importorskip("gdstk")

from squadds.layouts.dataset_export import (  # noqa: E402
    MARGIN_GAPS,
    dataset_export_settings,
    default_star_style,
    export_dataset_gds,
    family_of,
    generator_ground_gap_um,
)


def _component(cls_name, options=None):
    from qiskit_metal import designs

    from squadds.components import get_component_class

    design = designs.DesignPlanar({}, overwrite_enabled=True)
    comp = get_component_class(cls_name)(design, "c", options=options or {})
    design.rebuild()
    return comp


def _layers(path):
    out = {}
    for p in gdstk.read_gds(str(path)).top_level()[0].polygons:
        out.setdefault((p.layer, p.datatype), []).append(p)
    return out


def _chip_um(layers):
    (x0, y0), (x1, y1) = layers[(1, 0)][0].bounding_box()
    return (x1 - x0) / 1e3, (y1 - y0) / 1e3


def test_family_and_settings(headless_qiskit_environment):
    comp = _component("GeneralizedCapNInterdigital")
    assert family_of(comp) == family_of(comp.design) == family_of("GeneralizedCapNInterdigital") == "interdigital"
    assert family_of("TransmonCross") is None
    s = dataset_export_settings(comp)
    assert s["exporter"] == "helpers" and s["margin_gaps"] == MARGIN_GAPS["interdigital"] == 20.0
    assert s["export_style"] is None and s["ground_gap_um"] is None
    with pytest.raises(ValueError, match="star layouts only"):
        dataset_export_settings(comp, export_style="original")
    assert default_star_style({"pad_enables": "1,0,0,0,0"}) == "original"
    assert default_star_style({"pad_enables": "1,1,0,0,0"}) == "reexport_gdspy"


def test_families_ground_gap_rules(headless_qiskit_environment):
    xmon = _component("Xmon")
    s = dataset_export_settings(xmon)
    pad = xmon.options.connection_pads.readout
    expect = max(xmon.design.parse_value(xmon.options.cross_gap), xmon.design.parse_value(pad.ground_spacing)) * 1e3
    assert s["exporter"] == "families" and s["margin_gaps"] == 8.0 and s["ground_gap_um"] == pytest.approx(expect)
    assert generator_ground_gap_um("xmon", {"cross_gap": 30.0, "ground_spacing": 40.0}) == 40.0
    pocket = {"pocket_width": 650.0, "pad_width": 640.0, "pocket_height": 650.0, "pad_gap": 30.0, "pad_height": 90.0}
    assert generator_ground_gap_um("pocket", pocket) == 20.0  # floored
    assert generator_ground_gap_um("interdigital", {}) is None


def test_interdigital_layers_and_chip(headless_qiskit_environment, tmp_path):
    comp = _component("GeneralizedCapNInterdigital")
    path = export_dataset_gds(comp, tmp_path / "idc.gds")
    lay = _layers(path)
    assert {k: len(v) for k, v in lay.items()} == {(1, 10): 2, (1, 0): 1, (2, 0): 1, (3, 0): 1}
    d, o = comp.design, comp.options
    um = lambda k: float(d.parse_value(o[k])) * 1e3  # noqa: E731
    fc = int(d.parse_value(o.finger_count))
    cap_w = fc * um("finger_width") + (fc - 1) * um("finger_gap_east_west")
    cap_h = um("south_spine_width") + um("north_spine_width") + um("finger_length") + um("finger_gap_north_south")
    sx = 40 * max(um("west_gap_ground"), um("east_gap_ground")) + cap_w + um("west_gap_ground") + um("east_gap_ground")
    sy = 40 * max(um("north_gap_ground"), um("south_gap_ground")) + cap_h
    assert _chip_um(lay) == pytest.approx((sx, sy), abs=2e-3)
    port = lay[(2, 0)][0].bounding_box()
    assert (port[1][1] - port[0][1]) / 1e3 == pytest.approx(2.0)
    lib = gdstk.read_gds(str(path))
    assert lib.unit == pytest.approx(1e-9) and lib.precision == pytest.approx(1e-9)


def test_export_is_deterministic_and_position_free(headless_qiskit_environment, tmp_path):
    from squadds.ml.capacitance.qc import geom_hash

    a = export_dataset_gds(_component("GeneralizedCapConcentric"), tmp_path / "a.gds")
    b = export_dataset_gds(
        _component("GeneralizedCapConcentric", {"pos_x": "1.3mm", "pos_y": "-2mm", "orientation": "90"}),
        tmp_path / "b.gds",
    )
    assert geom_hash(a) == geom_hash(b)


def test_star_and_xmon_exports(headless_qiskit_environment, tmp_path):
    star = _component("GeneralizedCapStar", {"pad_enables": "1,0,0,0,0"})
    lay = _layers(export_dataset_gds(star, tmp_path / "star.gds"))
    assert len(lay[(1, 10)]) == 2 and len(lay[(1, 0)]) == 1 and len(lay[(2, 0)]) == 1
    o = star.options
    um = lambda k: float(star.design.parse_value(o[k])) * 1e3  # noqa: E731
    rp, g, cl = um("pocket_radius"), um("gap"), um("cpw_length")
    assert _chip_um(lay) == pytest.approx((2 * rp + 16 * g, 2 * rp + cl + 16 * g), abs=2e-3)
    reexp = _layers(export_dataset_gds(star, tmp_path / "star_re.gds", export_style="reexport"))
    assert len(reexp[(1, 10)]) == 2

    xmon = _component(
        "Xmon",
        {"connection_pads": {"readout": {}, "bus": {"connector_location": "270"}}},
    )
    p = export_dataset_gds(xmon, tmp_path / "xmon.gds")
    lay = _layers(p)
    assert len(lay[(1, 10)]) == 3 and len(lay[(2, 0)]) == 1 and (3, 0) not in lay
    wide = _layers(export_dataset_gds(xmon, tmp_path / "xmon_m.gds", margin_gaps=10))
    gap = dataset_export_settings(xmon)["ground_gap_um"]
    assert _chip_um(wide)[0] - _chip_um(lay)[0] == pytest.approx(4 * gap, abs=2e-3)


def test_not_a_dataset_family(headless_qiskit_environment, tmp_path):
    from qiskit_metal import designs
    from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross

    design = designs.DesignPlanar({}, overwrite_enabled=True)
    comp = TransmonCross(design, "q", options={})
    with pytest.raises(ValueError, match="not a dataset family"):
        export_dataset_gds(comp, tmp_path / "x.gds")
    TransmonCross(design, "q2", options={})
    with pytest.raises(ValueError, match="one component"):
        export_dataset_gds(design, tmp_path / "y.gds")


# Dataset records (geometry JSON + GDS, train split) bundled in tests/data; TLSQ_DATA_LOCAL points at a full copy.
DATA = Path(os.environ.get("TLSQ_DATA_LOCAL", Path(__file__).parent / "data" / "capml" / "designs"))
CASES = [
    ("interdigital", "designs_ood/data/design_files/exp6/q3d_cap_0602", "exp6"),
    ("concentric", "designs_ood/data/design_files/exp10/q3d_cap_0000", "exp10"),
    ("star", "designs_ood/data/design_files/exp11/q3d_cap_0001", "exp11"),
    ("xmon", "newfam/design_files/exp13/q3d_cap_0454", "exp13"),
    ("pocket", "newfam/design_files/exp14/q3d_cap_0153", "exp14"),
]


@pytest.mark.integration
@pytest.mark.parametrize("family,folder,exp", CASES)
def test_geom_hash_equals_the_dataset_gds(headless_qiskit_environment, tmp_path, family, folder, exp):
    """The export of a dataset record's component has the dataset GDS's geom_hash."""
    import importlib.util

    from squadds.ml.capacitance.qc import geom_hash

    folder = DATA / folder
    if not folder.is_dir():
        pytest.fail(f"dataset record missing: {folder}")
    script = Path(__file__).resolve().parents[1] / "scripts" / "planar_capacitance" / "verify_component_gds.py"
    spec = importlib.util.spec_from_file_location("verify_component_gds", script)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    tid = folder.name.split("_")[-1]
    cfg = json.loads((folder / f"cap_{tid}.json").read_text()) | {"exp": exp}
    comp, kw = verify.dataset_component(family, cfg)
    out = export_dataset_gds(comp, tmp_path / "out.gds", **kw)
    assert geom_hash(out) == geom_hash(folder / f"cap_{tid}.gds")
