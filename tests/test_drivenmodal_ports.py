from squadds.simulations.drivenmodal.ports import (
    build_capacitance_port_specs,
    build_coupled_system_port_specs,
    split_rendered_ports,
)


def test_build_capacitance_port_specs_for_ncap_has_expected_order():
    port_specs = build_capacitance_port_specs(
        "ncap",
        {
            "port_mapping": {
                "top": {"component": "NCapCoupler", "pin": "top"},
                "bottom": {"component": "NCapCoupler", "pin": "bottom"},
            }
        },
    )

    assert [spec.kind for spec in port_specs] == ["top", "bottom"]
    assert port_specs[0].to_qiskit_port_entry() == ("NCapCoupler", "top", 50.0)


def test_build_capacitance_port_specs_for_qubit_claw_marks_junction_target():
    port_specs = build_capacitance_port_specs(
        "qubit_claw",
        {
            "port_mapping": {
                "cross": {"component": "xmon", "pin": "rect_jj"},
                "claw": {"component": "xmon", "pin": "readout"},
            }
        },
    )

    port_list, jj_to_port = split_rendered_ports(port_specs)

    assert [spec.kind for spec in port_specs] == ["cross", "claw"]
    assert port_list == [("xmon", "readout", 50.0)]
    assert jj_to_port == [("xmon", "rect_jj", 50.0, False)]


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

    port_list, jj_to_port = split_rendered_ports(port_specs)

    assert [spec.kind for spec in port_specs] == ["feedline_input", "feedline_output", "jj"]
    assert port_list == [("ReadoutLine", "in", 50.0), ("ReadoutLine", "out", 50.0)]
    assert jj_to_port == [("Q1", "jj", 50.0, False)]
