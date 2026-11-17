import pytest

from squadds.simulations.sweeper_helperfunctions import (
    create_dict_list,
    expand_design_options,
    extract_parameters,
    extract_QSweep_parameters,
    extract_values,
    generate_combinations,
)


def test_extract_parameters_flattens_nested_keys_in_order():
    nested = {
        "qubit": {
            "cross_length": ["10um", "20um"],
            "connection_pads": {"readout": {"claw_length": ["30um", "40um"]}},
        },
        "coupler": {"finger_count": [1, 2]},
    }

    assert extract_parameters(nested) == [
        "qubit.cross_length",
        "qubit.connection_pads.readout.claw_length",
        "coupler.finger_count",
    ]


def test_extract_values_wraps_scalars_and_preserves_lists():
    nested = {
        "a": {"b": [1, 2], "c": {"d": "5um"}},
        "e": 3,
    }

    assert extract_values(nested) == [[1, 2], ["5um"], [3]]


def test_generate_combinations_returns_cartesian_product():
    combinations = generate_combinations([[1, 2], ["x", "y"]])

    assert combinations == [(1, "x"), (1, "y"), (2, "x"), (2, "y")]


def test_create_dict_list_rebuilds_nested_structure():
    keys = [
        "qubit.cross_length",
        "qubit.connection_pads.readout.claw_length",
        "coupler.finger_count",
    ]
    values = [
        ("10um", "30um", 1),
        ("20um", "40um", 2),
    ]

    assert create_dict_list(keys, values) == [
        {
            "qubit": {
                "cross_length": "10um",
                "connection_pads": {"readout": {"claw_length": "30um"}},
            },
            "coupler": {"finger_count": 1},
        },
        {
            "qubit": {
                "cross_length": "20um",
                "connection_pads": {"readout": {"claw_length": "40um"}},
            },
            "coupler": {"finger_count": 2},
        },
    ]


def test_extract_qsweep_parameters_builds_all_nested_combinations():
    sweep = {
        "cplr_opts": {"finger_count": [1, 2]},
        "claw_opts": {
            "connection_pads": {
                "readout": {
                    "claw_length": ["10um", "20um"],
                }
            }
        },
    }

    assert extract_QSweep_parameters(sweep) == [
        {
            "cplr_opts": {"finger_count": 1},
            "claw_opts": {"connection_pads": {"readout": {"claw_length": "10um"}}},
        },
        {
            "cplr_opts": {"finger_count": 1},
            "claw_opts": {"connection_pads": {"readout": {"claw_length": "20um"}}},
        },
        {
            "cplr_opts": {"finger_count": 2},
            "claw_opts": {"connection_pads": {"readout": {"claw_length": "10um"}}},
        },
        {
            "cplr_opts": {"finger_count": 2},
            "claw_opts": {"connection_pads": {"readout": {"claw_length": "20um"}}},
        },
    ]


def test_expand_design_options_builds_grid_of_list_valued_parameters():
    options = {
        "cross_length": ["200um", "250um"],
        "cross_gap": "30um",
        "connection_pads": {"readout": {"claw_length": ["150um", "200um", "250um"], "claw_gap": "5um"}},
        "notes": {},
    }

    designs = expand_design_options(options)

    assert len(designs) == 6
    assert designs[1] == {
        "cross_length": "200um",
        "cross_gap": "30um",
        "connection_pads": {"readout": {"claw_length": "200um", "claw_gap": "5um"}},
        "notes": {},
    }
    assert {(d["cross_length"], d["connection_pads"]["readout"]["claw_length"]) for d in designs} == {
        (c, w) for c in ("200um", "250um") for w in ("150um", "200um", "250um")
    }
    assert options["cross_length"] == ["200um", "250um"]


def test_expand_design_options_without_lists_returns_a_copy():
    options = {"cross_length": "200um", "jog": ("R90", "50um")}

    designs = expand_design_options(options)

    assert designs == [options]
    assert designs[0] is not options


def test_expand_design_options_sweeps_shared_parameter_together():
    claw = ["150um", "200um"]
    options = {
        "design_options_qubit": {"connection_pads": {"readout": {"claw_length": claw}}},
        "design_options_cavity_claw": {"claw_opts": {"connection_pads": {"readout": {"claw_length": list(claw)}}}},
    }

    designs = expand_design_options(options)

    assert len(designs) == 2
    for design in designs:
        qubit = design["design_options_qubit"]["connection_pads"]["readout"]["claw_length"]
        cavity = design["design_options_cavity_claw"]["claw_opts"]["connection_pads"]["readout"]["claw_length"]
        assert qubit == cavity


def test_expand_design_options_rejects_empty_list():
    with pytest.raises(ValueError, match="cross_length"):
        expand_design_options({"cross_length": []})
