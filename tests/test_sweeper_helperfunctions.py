from squadds.simulations.sweeper_helperfunctions import (
    create_dict_list,
    extract_QSweep_parameters,
    extract_parameters,
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
