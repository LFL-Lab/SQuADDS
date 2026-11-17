"""TransmonCross junction styles build in a headless design (#34)."""

import os

os.environ.setdefault("QISKIT_METAL_HEADLESS", "1")

import numpy as np
import pytest
from qiskit_metal import designs

from squadds.components.jjs import JjDolan, SquidLoopDolan
from squadds.components.qubits import TransmonCross

STYLES = {
    "JJ_Dolan": ("Dolan_JJ", "_JJ_Dolan", JjDolan, 1),
    "SQUID_LOOP_Dolan": ("SQUID_LOOP", "_SQUID_LOOP_Dolan", SquidLoopDolan, 2),
}


def _south_tip(orientation_deg, pos=(1.0, -0.5), arm=0.2, gap=0.02):
    """Centre of the gap at the end of the south arm, in mm, after rotation and translation."""
    theta = np.radians(orientation_deg)
    y = -arm - gap / 2
    return pos[0] - np.sin(theta) * y, pos[1] + np.cos(theta) * y


def _build(style, orientation="0", **extra):
    design = designs.DesignPlanar()
    options = dict(style=style, orientation=orientation, pos_x="1mm", pos_y="-0.5mm", **extra)
    qubit = TransmonCross(design, "Q1", options=options)
    return design, qubit


@pytest.mark.parametrize("style", sorted(STYLES))
@pytest.mark.parametrize("orientation", ["0", "90", "180"])
@pytest.mark.parametrize("flip", [False, True])
def test_dolan_styles_build_junctions_at_south_arm(style, orientation, flip):
    opts_key, suffix, cls, n_junctions = STYLES[style]
    design, qubit = _build(style, orientation, **{opts_key: {"JJ_flip": flip}})

    child_name = "Q1" + suffix
    assert child_name in design.name_to_id
    child = design.components[child_name]
    assert isinstance(child, cls)

    polys = design.qgeometry.tables["poly"]
    child_polys = polys[polys["component"] == child.id]
    assert len(child_polys) > 0
    assert all(g.is_valid and g.area > 0 for g in child_polys.geometry)
    assert (child_polys["layer"] == child.options.JJ_layer).sum() == n_junctions

    junctions = design.qgeometry.tables["junction"]
    child_junctions = junctions[junctions["component"] == child.id]
    assert len(child_junctions) == n_junctions

    qubit_polys = polys[polys["component"] == qubit.id]
    assert "T_pin2" in set(qubit_polys["name"])

    tip_x, tip_y = _south_tip(float(orientation))
    for geom in child_junctions.geometry:
        assert np.hypot(geom.centroid.x - tip_x, geom.centroid.y - tip_y) < 0.05


@pytest.mark.parametrize("style", sorted(STYLES))
def test_dolan_styles_survive_rebuild(style):
    opts_key, suffix, _, n_junctions = STYLES[style]
    design, qubit = _build(style, **{opts_key: {"JJ_width": "200nm"}})
    child = design.components["Q1" + suffix]

    design.rebuild()

    assert design.components["Q1" + suffix] is child
    junctions = design.qgeometry.tables["junction"]
    child_junctions = junctions[junctions["component"] == child.id]
    assert len(child_junctions) == n_junctions
    assert np.allclose(child_junctions["width"], 0.0002)


def test_default_style_junction_follows_orientation():
    design, qubit = _build("default", "90")
    junctions = design.qgeometry.tables["junction"]
    (geom,) = junctions[junctions["component"] == qubit.id].geometry
    tip_x, tip_y = _south_tip(90.0)
    assert np.hypot(geom.centroid.x - tip_x, geom.centroid.y - tip_y) < 1e-9


def test_unknown_style_is_rejected():
    design = designs.DesignPlanar()
    with pytest.raises(ValueError, match="SQUID_LOOP_Dolan"):
        TransmonCross(design, "Q1", options=dict(style="SquidLoopDolan"))
