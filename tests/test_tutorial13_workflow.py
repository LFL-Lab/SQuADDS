from __future__ import annotations

import importlib.util
from pathlib import Path


def load_tutorial13_module():
    root = Path(__file__).resolve().parents[1]
    tutorial13_path = root / "tutorials" / "Tutorial-13_DrivenModal_Combined_Hamiltonian_Extraction.py"
    spec = importlib.util.spec_from_file_location("tutorial13_dm", tutorial13_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Tutorial 13 module from {tutorial13_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_segmented_sweeps_covers_qubit_bridge_resonator_bands():
    module = load_tutorial13_module()
    reference_summary = {
        "qubit_frequency_ghz": 3.8877083222541025,
        "cavity_frequency_ghz": 8.96333323745,
    }

    sweeps = module.build_segmented_sweeps(reference_summary)

    assert set(sweeps) == {"qubit_band", "bridge_band", "resonator_band"}
    assert sweeps["qubit_band"].start_ghz < reference_summary["qubit_frequency_ghz"] < sweeps["qubit_band"].stop_ghz
    assert (
        sweeps["resonator_band"].start_ghz
        < reference_summary["cavity_frequency_ghz"]
        < sweeps["resonator_band"].stop_ghz
    )
    assert sweeps["bridge_band"].start_ghz == sweeps["qubit_band"].stop_ghz
    assert sweeps["bridge_band"].stop_ghz == sweeps["resonator_band"].start_ghz
    assert sweeps["bridge_band"].count == module.BRIDGE_BAND_COUNT


def test_build_combined_hamiltonian_table_includes_all_target_quantities():
    module = load_tutorial13_module()
    qubit_summary = {
        "selected_model_extracted": {
            "qubit_frequency_ghz": 3.74,
            "anharmonicity_mhz": -118.0,
        }
    }
    resonator_summary = {
        "reference": {
            "qubit_frequency_ghz": 3.8877083222541025,
            "anharmonicity_mhz": -128.7377085331389,
            "cavity_frequency_ghz": 8.96333323745,
            "kappa_mhz": 0.282985474218,
            "g_mhz": 52.30894220304142,
        },
        "extracted": {
            "cavity_frequency_ghz": 8.77,
            "kappa_mhz": 0.12,
            "chi_mhz": -0.045,
            "g_mhz": 59.8,
        },
    }

    table = module.build_combined_hamiltonian_table(qubit_summary, resonator_summary)

    assert list(table["quantity"]) == [
        "qubit_frequency_ghz",
        "anharmonicity_mhz",
        "cavity_frequency_ghz",
        "kappa_mhz",
        "g_mhz",
        "chi_mhz",
    ]
    assert table.iloc[0]["drivenmodal"] == 3.74
    assert table.iloc[-1]["reference"] != table.iloc[-1]["reference"]  # NaN
