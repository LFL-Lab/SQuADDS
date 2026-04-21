from __future__ import annotations

import importlib.util
from pathlib import Path


def load_tutorial12_module():
    root = Path(__file__).resolve().parents[1]
    tutorial12_path = root / "tutorials" / "Tutorial-12_DrivenModal_Qubit_Port_Admittance.py"
    spec = importlib.util.spec_from_file_location("tutorial12_dm", tutorial12_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Tutorial 12 module from {tutorial12_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_qubit_followup_frequency_prefers_linear_mode():
    module = load_tutorial12_module()

    summary = {
        "selected_model_extracted": {
            "linear_resonance_ghz": 4.123,
            "resonance_at_sweep_edge": False,
        }
    }

    assert module.select_qubit_followup_frequency(summary) == 4.123


def test_qubit_discovery_boundary_direction_detects_lower_and_upper_edges():
    module = load_tutorial12_module()
    sweep = module.T11.DrivenModalSweepSpec(
        name="DrivenModalSweep",
        start_ghz=3.0,
        stop_ghz=5.0,
        count=4000,
        sweep_type="Interpolating",
        save_fields=False,
        interpolation_tol=0.005,
        interpolation_max_solutions=400,
    )

    lower_summary = {
        "selected_model_extracted": {
            "linear_resonance_ghz": 3.0,
            "resonance_at_sweep_edge": True,
        }
    }
    upper_summary = {
        "selected_model_extracted": {
            "linear_resonance_ghz": 5.0,
            "resonance_at_sweep_edge": True,
        }
    }
    centered_summary = {
        "selected_model_extracted": {
            "linear_resonance_ghz": 4.2,
            "resonance_at_sweep_edge": False,
        }
    }

    assert module.qubit_discovery_boundary_direction(lower_summary, sweep) == "lower"
    assert module.qubit_discovery_boundary_direction(upper_summary, sweep) == "upper"
    assert module.qubit_discovery_boundary_direction(centered_summary, sweep) is None


def test_build_final_qubit_setup_and_sweep_recenters_window():
    module = load_tutorial12_module()

    setup, sweep = module.build_final_qubit_setup_and_sweep(4.35)

    assert setup.freq_ghz == 4.35
    assert sweep.start_ghz < 4.35 < sweep.stop_ghz
    assert sweep.count == module.QUBIT_FINAL_SWEEP_COUNT


def test_build_model_agreement_table_sorts_best_match_first():
    module = load_tutorial12_module()
    model_df = module.pd.DataFrame(
        [
            {
                "jj_capacitance_fF": 0.0,
                "jj_resistance_ohms": float("inf"),
                "qubit_frequency_ghz": 3.75,
                "anharmonicity_mhz": -119.0,
                "effective_capacitance_fF": 175.0,
            },
            {
                "jj_capacitance_fF": 2.0,
                "jj_resistance_ohms": 50_000.0,
                "qubit_frequency_ghz": 3.885,
                "anharmonicity_mhz": -128.2,
                "effective_capacitance_fF": 151.0,
            },
        ]
    )
    reference = {
        "qubit_frequency_ghz": 3.8877083222541025,
        "anharmonicity_mhz": -128.7377085331389,
        "alpha_hz": -128737708.53316557,
    }

    agreement_df = module.build_model_agreement_table(model_df, reference)

    assert agreement_df.iloc[0]["jj_capacitance_fF"] == 2.0
    assert agreement_df.iloc[0]["fq_abs_err_mhz"] < agreement_df.iloc[1]["fq_abs_err_mhz"]
    assert agreement_df.iloc[0]["cap_abs_err_fF"] < agreement_df.iloc[1]["cap_abs_err_fF"]
