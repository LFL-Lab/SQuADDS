import pandas as pd

from squadds.core.analysis_enrichment import (
    extract_coupler_options,
    extract_cpw_options,
    extract_qubit_options,
    fix_cavity_claw_dataframe,
)


def test_fix_cavity_claw_dataframe_renames_and_converts_units():
    df = pd.DataFrame(
        {
            "cavity_frequency": [7.2e9],
            "kappa": [350e3],
            "units": ["legacy"],
            "other": [1],
        }
    )

    fixed = fix_cavity_claw_dataframe(df)

    assert "cavity_frequency_GHz" in fixed.columns
    assert "kappa_kHz" in fixed.columns
    assert "units" not in fixed.columns
    assert fixed.loc[0, "cavity_frequency_GHz"] == 7.2
    assert fixed.loc[0, "kappa_kHz"] == 350.0
    assert fixed.loc[0, "other"] == 1


def test_extract_qubit_options_returns_expected_arrays():
    df = pd.DataFrame(
        {
            "design_options": [
                {
                    "qubit_options": {
                        "connection_pads": {
                            "readout": {
                                "claw_gap": "4um",
                                "claw_length": "30um",
                                "claw_width": "10um",
                                "ground_spacing": "6um",
                            }
                        },
                        "cross_gap": "20um",
                        "cross_length": "200um",
                        "cross_width": "18um",
                    }
                }
            ]
        }
    )

    extracted = extract_qubit_options(df)

    assert extracted["claw_gap"].tolist() == ["4um"]
    assert extracted["cross_length"].tolist() == ["200um"]


def test_extract_cpw_options_fills_none_for_invalid_rows():
    df = pd.DataFrame(
        {"design_options": [{"cavity_claw_options": {"cpw_opts": {"left_options": {"total_length": "4mm"}}}}]}
    )

    extracted = extract_cpw_options(df)

    assert extracted["total_length"].tolist() == [None]
    assert extracted["trace_gap"].tolist() == [None]
    assert extracted["trace_width"].tolist() == [None]


def test_extract_coupler_options_supports_clt_and_ncap_shapes():
    df = pd.DataFrame(
        {
            "design_options": [
                {
                    "cavity_claw_options": {
                        "coupler_type": "CLT",
                        "coupler_options": {
                            "coupling_length": "180um",
                            "coupling_space": "3um",
                            "down_length": "60um",
                            "orientation": "180",
                            "prime_gap": "6um",
                            "prime_width": "10um",
                            "second_gap": "6um",
                            "second_width": "10um",
                        },
                    }
                },
                {
                    "cavity_claw_options": {
                        "coupler_type": "NCap",
                        "coupler_options": {
                            "cap_distance": "75um",
                            "cap_gap": "4um",
                            "cap_gap_ground": "3um",
                            "cap_width": "14um",
                            "finger_count": 6,
                            "finger_length": "48um",
                            "orientation": "180",
                        },
                    }
                },
            ]
        }
    )

    extracted = extract_coupler_options(df)

    assert extracted["coupling_length"].tolist() == ["180um"]
    assert extracted["cap_distance"].tolist() == ["75um"]
    assert extracted["finger_count"].tolist() == [6]
    assert extracted["orientation"].tolist() == ["180", "180"]
