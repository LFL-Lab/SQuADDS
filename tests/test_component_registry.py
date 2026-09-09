import pytest
from qiskit_metal import Dict, designs
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

from squadds.components import (
    GeneralizedCapConcentric,
    GeneralizedCapNInterdigital,
    GeneralizedCapStar,
    build_component_from_design,
    get_component_class,
    get_coupler_spec,
)
from squadds.components.cavity_claw import CavityClaw
from squadds.components.coupled_systems import QubitCavity
from squadds.core.db import SQuADDS_DB


def test_generalized_dataset_row_builds_squadds_component():
    design = designs.DesignPlanar()
    row = {
        "design": {
            "component_module": "squadds.components",
            "component_class": "GeneralizedCapNInterdigital",
            "design_options": {"finger_count": "3"},
        }
    }

    component = build_component_from_design(design, row)

    assert type(component) is GeneralizedCapNInterdigital
    assert type(component).__module__ == "squadds.components.generalized_ncap_interdigital"
    assert sorted(component.pin_names) == ["north_end", "south_end"]


def test_flattened_generalized_api_row_builds_squadds_component():
    design = designs.DesignPlanar()
    row = {
        "component_class": "GeneralizedCapNInterdigital",
        "design_options": {"finger_count": "4"},
    }

    component = build_component_from_design(design, row)

    assert type(component) is GeneralizedCapNInterdigital


def test_database_api_builds_component_declared_by_row():
    design = designs.DesignPlanar()
    row = {
        "component_module": "squadds.components",
        "component_class": "GeneralizedCapNInterdigital",
        "design_options": {"finger_count": "2"},
    }

    component = SQuADDS_DB.build_qiskit_metal_component(design, row, name="from_api")

    assert type(component) is GeneralizedCapNInterdigital
    assert component.name == "from_api"


@pytest.mark.parametrize("wrapper_class", [CavityClaw, QubitCavity])
def test_modular_wrappers_use_squadds_generalized_component(wrapper_class):
    design = designs.DesignPlanar()
    options = Dict(
        cavity_claw_options=Dict(
            coupler_type="GeneralizedCapNInterdigital",
            coupler_options=Dict(finger_count="5", orientation="0"),
            cpw_opts=Dict(
                total_length="2000um",
                left_options=Dict(
                    trace_width="10um",
                    trace_gap="6um",
                    fillet="30um",
                    meander=Dict(spacing="100um"),
                ),
            ),
        ),
        qubit_options=Dict(connection_pads=Dict(readout=Dict(claw_cpw_width="10um"))),
    )

    if wrapper_class is QubitCavity:
        with pytest.warns(UserWarning, match="convenience wrapper"):
            wrapper = wrapper_class(design, "system", options=options)
    else:
        wrapper = wrapper_class(design, "system", options=options)

    assert type(wrapper.coupler) is GeneralizedCapNInterdigital
    assert wrapper.pin_names == {"north_end"}
    assert wrapper.LeftMeander.options.pin_inputs.end_pin.pin == "south_end"


def test_legacy_ncap_still_resolves_to_qiskit_metal():
    spec = get_coupler_spec("NCap")

    assert spec.component_class is CapNInterdigitalTee
    assert spec.route_pin == "second_end"
    assert spec.exposed_pins == ("prime_start", "prime_end")


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError, match="Unsupported coupler"):
        get_coupler_spec("unknown")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GeneralizedCapConcentric", GeneralizedCapConcentric),
        ("cap_concentric", GeneralizedCapConcentric),
        ("GeneralizedCapStar", GeneralizedCapStar),
        ("star-cap", GeneralizedCapStar),
    ],
)
def test_new_generalized_components_are_registered(name, expected):
    assert get_component_class(name) is expected


def test_concentric_dataset_row_builds_optional_two_terminal_component():
    design = designs.DesignPlanar()
    row = {
        "design": {
            "component_class": "GeneralizedCapConcentric",
            "design_options": {"inner_feed": True},
        }
    }

    component = build_component_from_design(design, row, name="concentric")

    assert type(component) is GeneralizedCapConcentric
    assert sorted(component.pin_names) == ["north_end", "south_end"]
    polygons = design.qgeometry.tables["poly"]
    polygons = polygons[polygons.component == component.id]
    assert set(polygons.name) == {"cap_etch", "cap_inner_body", "cap_outer_body"}
    assert all(geometry.is_valid for geometry in polygons.geometry)


def test_concentric_one_feed_row_omits_south_pin_like_exp10():
    design = designs.DesignPlanar()
    component = build_component_from_design(
        design,
        {
            "component_class": "GeneralizedCapConcentric",
            "design_options": {"inner_feed": False, "south_cpw_length": "0um"},
        },
    )

    assert sorted(component.pin_names) == ["north_end"]


def test_star_dataset_row_preserves_variable_pin_topology():
    design = designs.DesignPlanar()
    component = build_component_from_design(
        design,
        {
            "component_class": "GeneralizedCapStar",
            "design_options": {
                "n_points": "5",
                "pad_enables": "1,0,0,0,0",
                "valley_ground": "fill",
            },
        },
        name="star",
    )

    assert type(component) is GeneralizedCapStar
    assert sorted(component.pin_names) == ["pad_0"]
    polygons = design.qgeometry.tables["poly"]
    polygons = polygons[polygons.component == component.id]
    assert set(polygons.name) == {"cap_etch", "pad_0", "star"}
    assert all(geometry.is_valid for geometry in polygons.geometry)


def test_generalized_ncap_supports_asymmetric_terminal_widths_from_exp9():
    design = designs.DesignPlanar()
    component = GeneralizedCapNInterdigital(
        design,
        "asymmetric",
        options={
            "finger_count": "1",
            "finger_length": "0um",
            "finger_width": "10um",
            "north_finger_width": "30um",
            "south_finger_width": "12um",
        },
    )

    polygons = design.qgeometry.tables["poly"]
    polygons = polygons[polygons.component == component.id].set_index("name")
    north_width = polygons.loc["cap_north_body", "geometry"].bounds[2] - polygons.loc[
        "cap_north_body", "geometry"
    ].bounds[0]
    south_width = polygons.loc["cap_south_body", "geometry"].bounds[2] - polygons.loc[
        "cap_south_body", "geometry"
    ].bounds[0]

    assert north_width == pytest.approx(0.030)
    assert south_width == pytest.approx(0.012)
