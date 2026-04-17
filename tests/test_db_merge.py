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
