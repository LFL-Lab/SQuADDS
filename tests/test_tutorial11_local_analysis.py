from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_tutorial11_module():
    root = Path(__file__).resolve().parents[1]
    tutorial11_path = root / "tutorials" / "Tutorial-11_DrivenModal_Coupled_System_Postprocessing.py"
    spec = importlib.util.spec_from_file_location("tutorial11_dm", tutorial11_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Tutorial 11 module from {tutorial11_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_local_analysis_method_result_matches_chi_sign_convention():
    module = load_tutorial11_module()
    reference = {
        "f_q_hz": 3.887708322254131e9,
        "alpha_hz": -128.73770853316557e6,
    }
    result = module.build_local_analysis_method_result(
        8.775989817718987e9,
        8.775944361107323e9,
        reference,
    )

    assert result["f_ground_ghz"] == 8.775989817718987
    assert result["f_excited_ghz"] == 8.775944361107323
    assert result["chi_mhz"] < 0
    assert result["g_mhz"] > 0


def test_build_shift_threshold_rows_returns_first_crossing_per_capacitance():
    module = load_tutorial11_module()
    sweep_df = pd.DataFrame(
        [
            {"C_fF": 0.0, "L_nH": 0.1, "residual_feature_ghz": 8.0, "delta_vs_open_mhz": 0.01},
            {"C_fF": 0.0, "L_nH": 0.2, "residual_feature_ghz": 8.1, "delta_vs_open_mhz": 0.10},
            {"C_fF": 2.0, "L_nH": 0.1, "residual_feature_ghz": 8.2, "delta_vs_open_mhz": 0.20},
            {"C_fF": 2.0, "L_nH": 0.2, "residual_feature_ghz": 8.3, "delta_vs_open_mhz": 0.30},
        ]
    )

    thresholds = module.build_shift_threshold_rows(sweep_df, bin_hz=50_000.0)

    assert len(thresholds) == 2
    assert thresholds[0]["C_fF"] == 0.0
    assert thresholds[0]["first_visible_shift_L_nH"] == 0.2
    assert thresholds[1]["C_fF"] == 2.0
    assert thresholds[1]["first_visible_shift_L_nH"] == 0.1
