import json
import os
from pathlib import Path

import pytest
import shapely
from qiskit_metal import designs

from squadds.components import (
    GeneralizedCapConcentric,
    GeneralizedCapNInterdigital,
    GeneralizedCapStar,
    ShieldedCapConcentric,
    Xmon,
    build_component_from_design,
    get_component_class,
    get_coupler_spec,
)
from squadds.components.shielded_cap_concentric import shielded_options_from_generator

EXP6_CAP_0000 = {
    "north_cpw_length": "6um",
    "north_cpw_width": "2.2um",
    "north_cpw_gap": "2um",
    "north_cpw_radius": "1um",
    "north_cpw_etch_radius": "1um",
    "south_cpw_length": "6um",
    "south_cpw_width": "2.2um",
    "south_cpw_gap": "2um",
    "south_cpw_radius": "1um",
    "south_cpw_etch_radius": "1um",
    "north_spine_width": "4um",
    "north_west_spine_radius": "1um",
    "north_east_spine_radius": "1um",
    "north_west_spine_etch_radius": "1um",
    "north_east_spine_etch_radius": "1um",
    "south_spine_width": "4um",
    "south_west_spine_radius": "1um",
    "south_east_spine_radius": "1um",
    "south_west_spine_etch_radius": "1um",
    "south_east_spine_etch_radius": "1um",
    "north_gap_ground": "4um",
    "south_gap_ground": "4um",
    "east_gap_ground": "4um",
    "west_gap_ground": "4um",
    "finger_length": "8um",
    "finger_width": "4um",
    "finger_gap_north_south": "3um",
    "finger_gap_east_west": "3um",
    "finger_radius": "2.0um",
    "finger_etch_radius": "5.0um",
    "finger_edge_gap_radius": "1um",
    "finger_count": "2",
}

# area (um^2) and bounds (um) of every qgeometry drawn by the SQuADDS master (185f4e6) class
MASTER_INTERDIGITAL = {
    "default": (
        {},
        {
            "cap_etch": (5604.0, (-43.0, -86.0, 43.0, 0.0)),
            "cap_north_body": (1340.0, (-37.0, -50.0, 37.0, 0.0)),
            "cap_south_body": (1540.0, (-37.0, -86.0, 37.0, -36.0)),
        },
    ),
    "exp6_cap_0000": (
        EXP6_CAP_0000,
        {
            "cap_etch": (537.8, (-9.5, -31.0, 9.5, 0.0)),
            "cap_north_body": (92.654776, (-5.5, -18.0, 5.5, 0.0)),
            "cap_south_body": (92.654776, (-5.5, -31.0, 5.5, -13.0)),
        },
    ),
    "odd_offset_rotated": (
        {
            "finger_count": "5",
            "finger_length": "30um",
            "finger_width": "5um",
            "finger_gap_east_west": "4um",
            "finger_radius": "2um",
            "finger_etch_radius": "1.5um",
            "finger_edge_gap_radius": "1um",
            "north_cpw_xpos_offset": "3um",
            "south_cpw_xpos_offset": "-2um",
            "orientation": "90",
            "pos_x": "100um",
        },
        {
            "cap_etch": (4220.0, (100.0, -26.5, 196.0, 26.5)),
            "cap_north_body": (908.052958, (100.0, -20.5, 160.0, 20.5)),
            "cap_south_body": (1056.755666, (136.0, -20.5, 196.0, 20.5)),
        },
    ),
    "paddle_shared_width": (
        {
            "finger_count": "1",
            "finger_length": "0um",
            "finger_width": "40um",
            "north_spine_width": "20um",
            "south_spine_width": "30um",
            "finger_gap_north_south": "6um",
        },
        {
            "cap_etch": (4152.0, (-26.0, -96.0, 26.0, 0.0)),
            "cap_north_body": (1000.0, (-20.0, -40.0, 20.0, 0.0)),
            "cap_south_body": (1400.0, (-20.0, -96.0, 20.0, -46.0)),
        },
    ),
}


def _build(cls, options=None, name="c"):
    design = designs.DesignPlanar()
    component = cls(design, name, options=options or {})
    return design, component


def _geometry(design, subtract=None):
    table = design.qgeometry.tables["poly"]
    if subtract is not None:
        table = table[table["subtract"].astype(bool) == subtract]
    return dict(zip(table["name"], table.geometry))


def _assert_valid(design):
    geoms = _geometry(design)
    assert geoms
    for name, geom in geoms.items():
        assert geom.geom_type == "Polygon", name
        assert geom.is_valid and not geom.is_empty and geom.area > 0, name


def _metal_bodies(design):
    """Signal conductors: non-subtract polygons that touch merge into one body."""
    from shapely.ops import unary_union

    merged = unary_union(list(_geometry(design, subtract=False).values()))
    return len(getattr(merged, "geoms", [merged]))


@pytest.mark.parametrize("case", sorted(MASTER_INTERDIGITAL))
def test_interdigital_matches_master_geometry(case):
    options, expected = MASTER_INTERDIGITAL[case]
    design, _ = _build(GeneralizedCapNInterdigital, options)
    got = {
        name: (round(g.area * 1e6, 6), tuple(round(b * 1e3, 4) for b in g.bounds))
        for name, g in _geometry(design).items()
    }
    assert got == expected


def test_interdigital_side_finger_widths_fall_back_and_differ():
    base = {"finger_count": "1", "finger_length": "0um", "finger_width": "40um"}
    same, _ = _build(GeneralizedCapNInterdigital, base)
    explicit, _ = _build(GeneralizedCapNInterdigital, base | {"north_finger_width": "40um", "south_finger_width": ""})
    for name, geom in _geometry(same).items():
        assert geom.equals_exact(_geometry(explicit)[name], 0.0)

    paddle, component = _build(GeneralizedCapNInterdigital, base | {"north_finger_width": "60um"})
    geoms = _geometry(paddle)
    assert geoms["cap_north_body"].bounds[2] - geoms["cap_north_body"].bounds[0] == pytest.approx(0.060)
    assert geoms["cap_south_body"].bounds[2] - geoms["cap_south_body"].bounds[0] == pytest.approx(0.040)
    assert component.cap_width == pytest.approx(0.060)
    _assert_valid(paddle)
    assert _metal_bodies(paddle) == 2


def test_interdigital_tangent_ledge_rule():
    auto, _ = _build(GeneralizedCapNInterdigital, EXP6_CAP_0000)
    tangent, _ = _build(GeneralizedCapNInterdigital, EXP6_CAP_0000 | {"spine_ledge_rule": "tangent"})
    _assert_valid(tangent)
    assert _metal_bodies(tangent) == 2
    assert _geometry(tangent)["cap_etch"].equals_exact(_geometry(auto)["cap_etch"], 0.0)
    assert not _geometry(tangent)["cap_north_body"].equals_exact(_geometry(auto)["cap_north_body"], 0.0)
    with pytest.raises(ValueError, match="spine_ledge_rule"):
        _build(GeneralizedCapNInterdigital, {"spine_ledge_rule": "other"})


@pytest.mark.parametrize(
    ("options", "pins"),
    [({}, ["north_end", "south_end"]), ({"inner_feed": False, "south_cpw_length": "0um"}, ["north_end"])],
)
def test_concentric_defaults_and_feed_modes(options, pins):
    design, component = _build(GeneralizedCapConcentric, options)
    _assert_valid(design)
    assert sorted(component.pins) == pins
    assert _metal_bodies(design) == 2
    assert len(_geometry(design, subtract=True)) == 1


@pytest.mark.parametrize(("enables", "pads"), [("1,1,1,1,1", 5), ("1,0,1,0,0", 2), ("1,0,0,0,0", 1)])
def test_star_pads(enables, pads):
    design, component = _build(GeneralizedCapStar, {"pad_enables": enables, "valley_ground": "fill"})
    _assert_valid(design)
    assert len(component.pins) == pads
    assert _metal_bodies(design) == pads + 1


def test_star_export_helpers_or_group_keeps_duplicate():
    gdstk = pytest.importorskip("gdstk")
    from squadds.components.export_fixups import helpers_body_groups, star_dataset_bodies

    a = gdstk.rectangle((0, 0), (1e4, 1e4), layer=1, datatype=10)
    b = gdstk.rectangle((1e4, 1e4), (2e4, 2e4), layer=1, datatype=10)  # touches a at one corner only
    c = gdstk.rectangle((5e3, 5e3), (1.5e4, 1.5e4), layer=1, datatype=10)  # overlaps both
    groups = helpers_body_groups([a, b, c])
    assert len(groups) == 1 and [round(p.area()) for p in groups[0]] == [175e6, 175e6]
    bodies = star_dataset_bodies([a, b, c], "reexport")
    assert len(bodies) == 1 and round(bodies[0].area()) == 175e6  # b's far corner is dropped, as in the dataset
    assert round(star_dataset_bodies([a, b, c], "original")[0].area()) == 250e6
    with pytest.raises(ValueError, match="style"):
        star_dataset_bodies([a], "bogus")


def test_star_export_simplify_clockwise_keeps_first_vertex():
    import numpy as np

    gdstk = pytest.importorskip("gdstk")
    from squadds.components.export_fixups import simplify_clockwise

    pts = [(0, 0), (1000, 3), (2000, 0), (2000, 2000), (0, 2000)]  # counter-clockwise, (1000, 3) within 10 nm
    out = simplify_clockwise(gdstk.Polygon(pts))
    assert out.points[0].tolist() == [0, 0]
    assert len(out.points) == 4 and out.area() == 4e6
    assert (
        np.sum(out.points[:, 0] * np.roll(out.points[:, 1], -1) - np.roll(out.points[:, 0], -1) * out.points[:, 1]) < 0
    )


def test_star_export_isolates_lead_end_from_ground():
    gdstk = pytest.importorskip("gdstk")
    from squadds.components.export_fixups import star_dataset_ground

    ground = gdstk.boolean(
        gdstk.rectangle((-50000, -50000), (50000, 50000)),
        gdstk.rectangle((-11000, -20000), (11000, 0)),
        "not",
        layer=1,
        datatype=0,
    )[0]
    lead = gdstk.rectangle((-5000, -15000), (5000, 0), layer=1, datatype=10)  # ends on the cut-out edge y = 0
    out = star_dataset_ground(ground, [lead], "reexport")
    slit = ground.area() - out.area()
    assert slit == pytest.approx(10000 * 50 + 2 * 50 * 50 / 2, rel=1e-6)  # 50 nm deep, bevelled corners
    assert not gdstk.boolean(out, gdstk.rectangle((-5000, -15000), (5000, 50)), "and")
    assert star_dataset_ground(ground, [lead], "original") is ground


@pytest.mark.parametrize(("pads", "claws"), [({}, 1), ({"bus": {"connector_location": "270"}}, 2)])
def test_xmon_claws_and_no_junction(pads, claws):
    design, component = _build(Xmon, {"connection_pads": pads} if pads else {})
    _assert_valid(design)
    assert len(component.pins) == claws
    assert _metal_bodies(design) == claws + 1
    assert design.qgeometry.tables["junction"].empty


@pytest.mark.parametrize(
    ("options", "metal", "bodies"),
    [
        ({}, {"inner", "outer_ring", "guard_1"}, 3),
        (
            {"guard_gaps": "10um,8um", "guard_widths": "8um", "guard_grounded": "1,0"},
            {"inner", "outer_ring", "guard_1"},
            3,
        ),
        ({"guard_gaps": "10um,8um", "guard_widths": "8um,6um", "guard_grounded": "0,0"}, None, 4),
        ({"guard_grounded": "1"}, {"inner", "outer_ring"}, 2),
    ],
)
def test_shielded_concentric_guards(options, metal, bodies):
    design, component = _build(ShieldedCapConcentric, options)
    _assert_valid(design)
    assert list(component.pins) == ["readout"]
    if metal is not None:
        assert set(_geometry(design, subtract=False)) == metal
    assert _metal_bodies(design) == bodies


def test_shielded_options_from_generator_round_trip():
    params = {
        "r_s": 43.866112812552714,
        "aspect": 1.25,
        "guards": [{"gap": 20.7, "width": 24.8, "grounded": True}, {"gap": 5.0, "width": 7.5, "grounded": False}],
        "gap_n": 4.01,
        "w_n": 49.2,
        "gap_g": 29.87,
        "strap_w": 15.1,
        "strap_gap": 9.39,
        "cpw_len": 20.0,
    }
    options = shielded_options_from_generator(params)
    assert options["guard_grounded"] == "1,0"
    design, component = _build(ShieldedCapConcentric, options)
    _assert_valid(design)
    r_pocket = 43.866112812552714 + 20.7 + 24.8 + 5.0 + 7.5 + 4.01 + 49.2 + 29.87
    assert component.pins["readout"]["middle"][1] * 1e3 == pytest.approx(r_pocket + 20.0)


def test_registry_resolves_dataset_families():
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

    assert get_component_class("GeneralizedCapConcentric") is GeneralizedCapConcentric
    assert get_component_class("GeneralizedCapStar") is GeneralizedCapStar
    assert get_component_class("ShieldedCapConcentric") is ShieldedCapConcentric
    assert get_component_class("Xmon") is Xmon
    assert get_component_class("TransmonPocket") is TransmonPocket
    assert get_coupler_spec("GeneralizedCapConcentric").route_pin == "south_end"
    design = designs.DesignPlanar()
    row = {"design": {"component_class": "GeneralizedCapStar", "design_options": {"pad_enables": "1,1,0,0,0"}}}
    assert sorted(build_component_from_design(design, row).pins) == ["pad_0", "pad_1"]
    with pytest.raises(ValueError, match="Unsupported component"):
        get_component_class("NotAComponent")


# Dataset records (geometry JSON + GDS, train split) bundled in tests/data; TLSQ_DATA_ROOT points at a full copy.
DATA_ROOT = Path(os.environ.get("TLSQ_DATA_ROOT", Path(__file__).parent / "data" / "capml" / "designs"))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("family", "exp", "tid"),
    [
        ("interdigital", "exp6", "0000"),
        ("interdigital", "exp8", "00001"),
        ("paddle", "exp9", "00000"),
        ("concentric", "exp10", "0000"),
        ("star", "exp11", "0002"),
        ("star", "exp11", "1169"),
        ("multipad_star", "exp12", "0014"),
        ("xmon", "exp13", "0414"),
        ("pocket", "exp14", "0154"),
        ("shielded_concentric", "exp15", "0029"),
    ],
)
def test_dataset_gds_rebuilds_exactly(family, exp, tid, tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "planar_capacitance" / "verify_component_gds.py"
    sub = "designs_ood/data/design_files" if int(exp[3:]) <= 12 else "newfam/design_files"
    folder = DATA_ROOT / sub / exp / f"q3d_cap_{tid}"
    if not folder.is_dir():
        pytest.fail(f"dataset record missing: {folder}")
    spec = importlib.util.spec_from_file_location("verify_component_gds", script)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    if exp == "exp12":
        pytest.importorskip("gdspy")
    record = verify.check_one((family, exp, str(folder), str(tmp_path)))
    assert "error" not in record, record.get("traceback")
    assert record["conductors_ok"]
    if exp == "exp12" and shapely.geos_version >= (3, 12, 0):
        # exp12 was drawn under GEOS 3.11: newer GEOS moves a few 1 nm vertices (see squadds.components.export_fixups)
        assert record["layers"]["1/0"]["hash_equal"] and record["max_xor_um2"] < 1.0
    else:
        assert record["exact"], json.dumps(record["layers"])
