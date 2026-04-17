import json

import pandas as pd

from squadds.interpolations.physics import ScalingInterpolator


class DummyAnalyzer:
    def __init__(self, qubit_df, cavity_df):
        self.df = qubit_df.copy()
        self.selected_resonator_type = "quarter"
        self.selected_system = ["qubit", "cavity_claw"]
        self.H_param_keys = ["qubit_frequency_GHz", "anharmonicity_MHz", "g_MHz"]
        self.selected_coupler = "CLT"
        self._qubit_df = qubit_df
        self._cavity_df = cavity_df

    def find_closest(self, target_params, **kwargs):
        if "qubit_frequency_GHz" in target_params:
            return self._qubit_df.copy()
        return self._cavity_df.copy()


def test_scaling_interpolator_accepts_json_string_design_payloads(monkeypatch):
    monkeypatch.setattr(
        "squadds.interpolations.physics.Convert.Lj_from_Ej",
        lambda value, units_in, units_out: 8.0,
    )

    qubit_df = pd.DataFrame(
        {
            "qubit_frequency_GHz": [4.5],
            "anharmonicity_MHz": [-200.0],
            "g_MHz": [70.0],
            "cross_to_claw": [12.0],
            "EJ": [20.0],
            "design_options_qubit": [
                json.dumps(
                    {
                        "cross_length": "200um",
                        "aedt_hfss_inductance": 0.0,
                        "aedt_q3d_inductance": 0.0,
                        "q3d_inductance": 0.0,
                        "hfss_inductance": 0.0,
                        "connection_pads": {
                            "readout": {
                                "claw_length": "150um",
                                "ground_spacing": "20um",
                                "claw_cpw_length": "10um",
                            }
                        },
                    }
                )
            ],
            "setup_qubit": [{"setup": {"max_passes": 10}}],
        }
    )
    cavity_df = pd.DataFrame(
        {
            "cross_to_claw": [12.0],
            "cavity_frequency_GHz": [7.0],
            "kappa_kHz": [120.0],
            "design_options_cavity_claw": [
                {
                    "cpw_opts": {"total_length": "4000um"},
                    "cplr_opts": {"coupling_length": "250um"},
                    "claw_opts": {"connection_pads": {"readout": {"ground_spacing": "20um"}}},
                }
            ],
            "setup_cavity_claw": [{"setup": {"max_passes": 12}}],
        }
    )

    analyzer = DummyAnalyzer(qubit_df, cavity_df)
    design_df = ScalingInterpolator(
        analyzer,
        {
            "qubit_frequency_GHz": 4.5,
            "anharmonicity_MHz": -200.0,
            "g_MHz": 70.0,
            "cavity_frequency_GHz": 7.0,
            "kappa_kHz": 120.0,
            "resonator_type": "quarter",
        },
    ).get_design()

    row = design_df.iloc[0]
    assert row["design_options_qubit"]["cross_length"] == "200.0um"
    assert row["design_options_qubit"]["connection_pads"]["readout"]["claw_length"] == "150.0um"
    assert row["design_options_qubit"]["connection_pads"]["readout"]["claw_cpw_length"] == "0um"
    assert row["setup_qubit"] == {"max_passes": 10}
    assert row["setup_cavity_claw"] == {"max_passes": 12}
