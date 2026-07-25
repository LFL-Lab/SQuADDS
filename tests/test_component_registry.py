import pytest
from qiskit_metal import Dict, designs
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

from squadds.components import GeneralizedCapNInterdigital, build_component_from_design, get_coupler_spec
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
