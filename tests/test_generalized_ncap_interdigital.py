from qiskit_metal import designs
from shapely.geometry import Polygon

from squadds.components import GeneralizedCapNInterdigital


def build_component(options=None):
    design = designs.DesignPlanar()
    component = GeneralizedCapNInterdigital(design, "cplr", options=options or {})
    return design, component


def test_generalized_ncap_is_exported_and_builds_default_geometry():
    design, component = build_component()

    assert component.name == "cplr"
    assert sorted(component.pin_names) == ["north_end", "south_end"]
    assert len(design.qgeometry.tables["poly"]) == 3

    for geometry in design.qgeometry.tables["poly"]["geometry"]:
        assert geometry.is_valid
        assert isinstance(geometry, Polygon)


def test_generalized_ncap_supports_odd_and_even_finger_counts():
    for finger_count in ("1", "5", "6"):
        design, component = build_component({"finger_count": finger_count})

        assert component.pin_names
        assert len(design.qgeometry.tables["poly"]) == 3


def test_generalized_ncap_honors_asymmetric_cpw_lengths():
    design, component = build_component(
        {
            "north_cpw_length": "10um",
            "south_cpw_length": "20um",
        }
    )

    assert sorted(component.pin_names) == ["north_end", "south_end"]
    assert len(design.qgeometry.tables["poly"]) == 3
