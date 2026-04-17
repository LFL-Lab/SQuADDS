from squadds.simulations.drivenmodal.ports import (
    build_capacitance_port_specs,
    build_coupled_system_port_specs,
)


def test_build_capacitance_port_specs_for_ncap_has_expected_order():
    port_specs = build_capacitance_port_specs(
        "ncap",
        {
            "port_mapping": {
                "top": {"component": "NCapCoupler", "pin": "top"},
                "bottom": {"component": "NCapCoupler", "pin": "bottom"},
                "ground": {"component": "chip_ground", "pin": "ground_ref"},
            }
        },
    )

    assert [spec.kind for spec in port_specs] == ["top", "bottom", "ground"]
    assert port_specs[0].to_qiskit_port_entry() == ("NCapCoupler", "top", 50.0)


def test_build_coupled_system_port_specs_has_feedline_and_jj_ports():
    port_specs = build_coupled_system_port_specs(
        {
            "port_mapping": {
                "feedline_input": {"component": "ReadoutLine", "pin": "in"},
                "feedline_output": {"component": "ReadoutLine", "pin": "out"},
                "jj": {"component": "Q1", "pin": "jj"},
            }
        },
    )

    assert [spec.kind for spec in port_specs] == ["feedline_input", "feedline_output", "jj"]
    assert port_specs[-1].to_qiskit_port_entry() == ("Q1", "jj", 50.0)
