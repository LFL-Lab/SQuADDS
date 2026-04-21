import pandas as pd

from squadds.core.db_merge import add_merger_terms_columns, create_qubit_cavity_dataframe


def _build_qubit_df():
    return pd.DataFrame(
        {
            "design_options": [
                {
                    "connection_pads": {
                        "readout": {
                            "claw_length": "30um",
                            "claw_gap": "4um",
                            "ground_spacing": "6um",
                        }
                    },
                    "cross_length": "200um",
                    "cross_gap": "20um",
                    "cross_width": "18um",
                }
            ],
            "qubit_frequency_GHz": [5.1],
        }
    )


def _build_cavity_df():
    return pd.DataFrame(
        {
            "design_options": [
                {
                    "claw_opts": {
                        "connection_pads": {
                            "readout": {
                                "claw_length": "30um",
                                "ground_spacing": "10um",
                            }
                        }
                    },
                    "cplr_opts": {"orientation": "180"},
                    "cpw_opts": {"total_length": "4000um", "trace_gap": "6um", "trace_width": "10um"},
                }
            ],
            "coupler_type": ["CLT"],
            "cavity_frequency_GHz": [7.0],
        }
    )


def test_add_merger_terms_columns_extracts_nested_lookup_values():
    qubit_df, cavity_df = add_merger_terms_columns(_build_qubit_df(), _build_cavity_df(), ["claw_length"])

    assert qubit_df["claw_length"].tolist() == ["30um"]
    assert cavity_df["claw_length"].tolist() == ["30um"]


def test_create_qubit_cavity_dataframe_merges_rows_and_builds_unified_design_options():
    merged_df = create_qubit_cavity_dataframe(
        _build_qubit_df(),
        _build_cavity_df(),
        merger_terms=["claw_length"],
        parallelize=False,
    )

    assert merged_df["index_qc"].tolist() == [0]
    assert merged_df["claw_length"].tolist() == ["30um"]
    assert "design_options" in merged_df.columns
    assert merged_df.iloc[0]["design_options"]["cavity_claw_options"]["coupler_type"] == "CLT"
    assert merged_df.iloc[0]["design_options"]["qubit_options"]["cross_length"] == "200um"


def test_create_qubit_cavity_dataframe_accepts_none_merger_terms():
    merged_df = create_qubit_cavity_dataframe(
        _build_qubit_df(),
        _build_cavity_df(),
        merger_terms=None,
        parallelize=False,
    )

    assert merged_df["index_qc"].tolist() == [0]
    assert "design_options" in merged_df.columns


def _build_cavity_df_with_json_string_payloads():
    # Mirrors the HuggingFace dataset schema where nested sub-payloads such as
    # `cplr_opts`, `lead`, and `meander` arrive as JSON strings rather than dicts.
    return pd.DataFrame(
        {
            "design_options": [
                {
                    "claw_opts": {
                        "connection_pads": {
                            "readout": {
                                "claw_length": "30um",
                                "ground_spacing": "10um",
                            }
                        }
                    },
                    "cplr_opts": '{"prime_width":"11.7um","prime_gap":"5.1um","coupling_length":"100um"}',
                    "cpw_opts": {
                        "total_length": "4000um",
                        "trace_gap": "6um",
                        "trace_width": "10um",
                        "lead": '{"start_straight":"50um"}',
                        "meander": '{"spacing":"100um","asymmetry":"-50.0um"}',
                    },
                }
            ],
            "coupler_type": ["CLT"],
            "cavity_frequency_GHz": [7.0],
        }
    )


def test_create_qubit_cavity_dataframe_inflates_json_string_subpayloads():
    # Regression test: downstream ML/tutorial workflows mutate
    # `design_options_cavity_claw` via nested dict access, so any JSON-string
    # sub-payloads in the upstream dataset must be inflated to real dicts.
    merged_df = create_qubit_cavity_dataframe(
        _build_qubit_df(),
        _build_cavity_df_with_json_string_payloads(),
        merger_terms=["claw_length"],
        parallelize=False,
    )

    cavity_payload = merged_df.iloc[0]["design_options_cavity_claw"]
    assert isinstance(cavity_payload["cplr_opts"], dict)
    assert cavity_payload["cplr_opts"]["prime_width"] == "11.7um"
    assert isinstance(cavity_payload["cpw_opts"]["lead"], dict)
    assert cavity_payload["cpw_opts"]["lead"]["start_straight"] == "50um"
    assert isinstance(cavity_payload["cpw_opts"]["meander"], dict)
    assert cavity_payload["cpw_opts"]["meander"]["spacing"] == "100um"

    # The unified design_options column should be consistent with the inflated
    # source columns (no lingering JSON strings to trip up consumers).
    unified = merged_df.iloc[0]["design_options"]
    assert isinstance(unified["cavity_claw_options"]["coupler_options"], dict)
    assert unified["cavity_claw_options"]["coupler_options"]["prime_width"] == "11.7um"
    assert isinstance(unified["cavity_claw_options"]["cpw_opts"]["left_options"]["meander"], dict)
