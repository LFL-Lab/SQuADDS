import json

import pytest

from squadds.database.generalized_ncap_dataset import (
    COMPONENT_NAME,
    CONTRIBUTOR,
    convert_dataset,
    convert_source_record,
)


def source_record():
    return {
        "cap_instance_name": "cap_0001",
        "margin_factor_ew": 20,
        "margin_factor_ns": 20,
        "layer_stack": {"metal_thickness_um": 0.25},
        "cap_options": {"finger_count": "5", "north_cpw_width": "10um"},
        "gds_north_port_length": "2um",
        "gds_north_port_layer": "2",
        "gds_south_port_length": "2um",
        "gds_south_port_layer": "3",
        "mesh": {"q3d_signal": {}},
        "ansys_gds": {"q3d_signal": {"setup": {"AdaptiveFreq": "13GHz"}}},
        "cap_matrix": {
            "matrix_elements_fF": {
                "C_G_G": 50.0,
                "C_G_N": -12.0,
                "C_G_S": -14.0,
                "C_N_G": -12.0,
                "C_N_N": 22.0,
                "C_N_S": -9.0,
                "C_S_G": -14.0,
                "C_S_N": -9.0,
                "C_S_S": 24.0,
            }
        },
    }


def test_convert_source_record_uses_canonical_names_and_contributor():
    row = convert_source_record(source_record(), "exp6", "exp6/cap_0001.json")

    assert row["contributor"] == CONTRIBUTOR
    assert row["design"]["coupler_type"] == COMPONENT_NAME
    assert row["design"]["component_module"] == "squadds.components"
    assert row["design"]["pin_aliases"] == {"top": "north_end", "bottom": "south_end"}
    assert row["notes"]["source_id"] == "exp6/cap_0001"
    assert row["sim_results"]["north_to_south"] == 9.0
    assert row["sim_results"]["top_to_bottom"] == 9.0
    assert row["sim_results"]["maxwell_matrix"]["C_N_S"] == -9.0


def test_convert_dataset_orders_rows_and_keeps_campaign_ids_unique(tmp_path):
    source_root = tmp_path / "export1"
    for campaign in ("exp7", "exp6"):
        directory = source_root / campaign
        directory.mkdir(parents=True)
        with (directory / "cap_0001.json").open("w") as destination:
            json.dump(source_record(), destination)

    output = tmp_path / "dataset.json"
    count = convert_dataset(source_root, output)
    rows = json.loads(output.read_text())

    assert count == 2
    assert [row["notes"]["source_id"] for row in rows] == ["exp6/cap_0001", "exp7/cap_0001"]


def test_convert_source_record_rejects_asymmetric_matrix():
    record = source_record()
    record["cap_matrix"]["matrix_elements_fF"]["C_S_N"] = -8.0

    with pytest.raises(ValueError, match="not symmetric"):
        convert_source_record(record, "exp6", "exp6/cap_0001.json")
