import json

from squadds.simulations.utils import (
    calculate_center_and_dimensions,
    chunk_sweep_options,
    extract_value,
    flatten_dict,
    get_cavity_claw_options_keys,
    read_json_files,
)


def test_get_cavity_claw_options_keys_matches_standard_prefixes():
    cpw_key, cplr_key = get_cavity_claw_options_keys({"cpw_opts": {}, "cplr_opts": {}})
    assert cpw_key == "cpw_opts"
    assert cplr_key == "cplr_opts"


def test_calculate_center_and_dimensions_returns_expected_tuple():
    center, dimensions = calculate_center_and_dimensions({"min_x": 0, "max_x": 10, "min_y": 2, "max_y": 6})
    assert center == (5.0, 4.0, 0)
    assert dimensions == (10, 4, 0)


def test_chunk_sweep_options_splits_claw_lengths_across_chunks():
    chunks = chunk_sweep_options(
        {
            "claw_opts": {"connection_pads": {"readout": {"claw_length": [10, 20, 30, 40]}}},
            "cpw_opts": {"total_length": [1000, 2000]},
            "cplr_opts": {"finger_count": [3]},
        },
        2,
    )

    assert chunks[0]["claw_opts"]["connection_pads"]["readout"]["claw_length"] == [10, 20]
    assert chunks[1]["claw_opts"]["connection_pads"]["readout"]["claw_length"] == [30, 40]
    assert chunks[0]["cpw_opts"]["total_length"] == [1000, 2000]


def test_extract_value_and_flatten_dict_handle_nested_mappings():
    nested = {"a": {"b": {"c": "12um"}}}
    assert extract_value(nested, "c") == "12um"
    assert flatten_dict({"a": {"b": 1}}) == {"a,b": 1}


def test_read_json_files_loads_all_json_payloads(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps({"value": 1}))
    (tmp_path / "two.json").write_text(json.dumps({"value": 2}))

    values = sorted(item["value"] for item in read_json_files(str(tmp_path)))

    assert values == [1, 2]
