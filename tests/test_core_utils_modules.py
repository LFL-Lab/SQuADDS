import numpy as np
import pandas as pd

from squadds.core.utils import (
    create_mailto_link,
    create_unified_design_options,
    filter_df_by_conditions,
    flatten_df_second_level,
    get_config_schema,
    get_sim_results_keys,
    get_type,
)


def test_get_type_matches_legacy_top_level_rules():
    assert get_type({"a": 1}) == "dict"
    assert get_type([1.0, 2.0]) == "float"


def test_get_config_schema_expands_only_supported_levels():
    schema = get_config_schema(
        {
            "sim_results": {"g_MHz": 55.0},
            "design": {"design_options": {"cross_length": "120um"}, "design_tool": "qm"},
            "notes": {"source": "paper"},
        }
    )

    assert schema == {
        "sim_results": {"g_MHz": "float"},
        "design": {"design_options": "dict", "design_tool": "str"},
        "notes": {"source": "str"},
    }


def test_flatten_df_second_level_and_get_sim_results_keys_preserve_legacy_shape():
    df = pd.DataFrame({"sim_results": [{"g_MHz": 55.0, "kappa_kHz": 120.0}], "meta": [1]})

    flattened = flatten_df_second_level(df)

    assert flattened.columns.tolist() == ["g_MHz", "kappa_kHz", "meta"]
    assert sorted(get_sim_results_keys(df)) == ["g_MHz", "kappa_kHz"]


def test_filter_df_by_conditions_returns_original_df_when_filter_is_empty():
    df = pd.DataFrame({"coupler_type": ["CLT"], "value": [1]})

    filtered = filter_df_by_conditions(df, {"coupler_type": "NCap"})

    assert filtered.equals(df)


def test_create_mailto_link_encodes_spaces_as_percent_20():
    link = create_mailto_link(["a@example.com"], "hello world", "body text")
    assert link == "mailto:a@example.com?subject=hello%20world&body=body%20text"


def test_create_unified_design_options_builds_expected_nested_payload():
    row = {
        "design_options_cavity_claw": {
            "cplr_opts": {"finger_count": 5},
            "cpw_opts": {"total_length": "1000um"},
            "claw_opts": {"connection_pads": {"readout": {"claw_cpw_width": "5um", "claw_cpw_length": "7um", "ground_spacing": "9um"}}},
        },
        "design_options_qubit": {
            "cross_length": "120um",
            "connection_pads": {"readout": {"claw_cpw_width": "1um", "claw_cpw_length": "2um", "ground_spacing": "3um"}},
        },
        "coupler_type": "NCap",
    }

    unified = create_unified_design_options(row)

    assert unified["cavity_claw_options"]["coupler_type"] == "NCap"
    assert unified["cavity_claw_options"]["coupler_options"] == {"finger_count": 5}
    assert unified["qubit_options"]["connection_pads"]["readout"]["claw_cpw_width"] == "0um"
    assert unified["cavity_claw_options"]["cpw_opts"]["left_options"] == {"total_length": "1000um"}
