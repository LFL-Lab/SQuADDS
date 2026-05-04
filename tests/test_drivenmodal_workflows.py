from __future__ import annotations

import math

import pandas as pd
import pytest

from squadds.simulations.drivenmodal import (
    build_capacitance_request,
    build_coupled_system_request,
    build_segmented_coupled_system_requests,
    capacitance_comparison_table,
    capacitance_reference_summary,
    default_hamiltonian_setup,
    hamiltonian_comparison_table,
    maxwell_matrix_interpretation,
    segmented_hamiltonian_sweeps,
)


def test_build_capacitance_request_uses_standard_qubit_claw_ports():
    row = {
        "design": {"design_options": {"cross_width": "30um"}},
        "sim_results": {
            "cross_to_ground": 80.0,
            "claw_to_ground": 20.0,
            "cross_to_claw": 5.0,
            "cross_to_cross": 85.0,
            "claw_to_claw": 25.0,
            "ground_to_ground": 105.0,
        },
    }

    request = build_capacitance_request(row, system_kind="qubit_claw", run_id="unit-qubit")

    assert request.system_kind == "qubit_claw"
    assert request.metadata["run_id"] == "unit-qubit"
    assert request.design_payload["port_mapping"]["cross"]["pin"] == "rect_jj"
    assert request.design_payload["port_mapping"]["claw"]["pin"] == "readout"
    assert capacitance_reference_summary(row, system_kind="qubit_claw")["cross_to_claw"] == 5.0


def test_capacitance_reference_summary_accepts_flattened_squadds_rows():
    row = {
        "cross_to_ground": 80.0,
        "claw_to_ground": 20.0,
        "cross_to_claw": 5.0,
        "cross_to_cross": 85.0,
        "claw_to_claw": 25.0,
        "ground_to_ground": 105.0,
    }

    assert capacitance_reference_summary(row, system_kind="qubit_claw")["cross_to_ground"] == 80.0


def test_capacitance_comparison_table_reports_percent_error_without_double_counting():
    table = capacitance_comparison_table(
        drivenmodal_fF={"cross_to_ground": 82.0, "cross_to_claw": 4.0},
        q3d_fF={"cross_to_ground": 80.0, "cross_to_claw": 5.0},
    )

    assert list(table["quantity"]) == ["cross_to_ground", "cross_to_claw"]
    assert table.loc[0, "percent_error"] == 2.5
    assert "cross_to_ground + cross_to_claw" in set(maxwell_matrix_interpretation()["entry"])


def test_build_coupled_system_request_normalizes_split_payload_rows():
    row = pd.Series(
        {
            "coupler_type": "CLT",
            "design_options_qubit": {
                "aedt_q3d_inductance": 10.0,
                "connection_pads": {"readout": {"claw_cpw_length": "40um"}},
            },
            "design_options_cavity_claw": {
                "coupler_options": {"coupling_length": "200um"},
                "cpw_options": {"total_length": "5000um"},
            },
        }
    )

    setup = default_hamiltonian_setup(freq_ghz=7.0)
    sweep = segmented_hamiltonian_sweeps(
        {"qubit_frequency_ghz": 4.0, "cavity_frequency_ghz": 7.0},
        qubit_count=101,
        bridge_count=21,
        resonator_count=101,
    )["resonator_band"]
    request = build_coupled_system_request(
        row,
        resonator_type="quarter",
        run_id="unit-coupled",
        setup=setup,
        sweep=sweep,
    )

    assert request.resonator_type == "quarter_wave"
    assert request.design_payload["port_mapping"]["feedline_input"]["component"] == "feedline"
    assert request.design_payload["port_mapping"]["jj"]["pin"] == "rect_jj"
    assert "left_options" in request.design_payload["design_options"]["cavity_claw_options"]["cpw_opts"]


def test_segmented_hamiltonian_sweeps_cover_qubit_bridge_and_resonator_windows():
    sweeps = segmented_hamiltonian_sweeps(
        {"qubit_frequency_ghz": 4.0, "cavity_frequency_ghz": 8.0},
        qubit_count=101,
        bridge_count=21,
        resonator_count=101,
    )

    assert list(sweeps) == ["qubit_band", "bridge_band", "resonator_band"]
    assert sweeps["qubit_band"].start_ghz < 4.0 < sweeps["qubit_band"].stop_ghz
    assert sweeps["resonator_band"].start_ghz < 8.0 < sweeps["resonator_band"].stop_ghz
    assert sweeps["bridge_band"].sweep_type == "Fast"


def test_default_hamiltonian_setup_uses_validated_drivenmodal_settings():
    setup = default_hamiltonian_setup(freq_ghz=8.0)

    assert setup.max_delta_s == 0.005
    assert setup.min_converged == 7
    assert setup.basis_order == -1


def test_build_segmented_coupled_system_requests_share_setup_and_use_named_sweeps():
    row = pd.Series(
        {
            "coupler_type": "CLT",
            "design_options_qubit": {"aedt_q3d_inductance": 10.0},
            "design_options_cavity_claw": {
                "coupler_options": {},
                "cpw_options": {"total_length": "5000um"},
            },
        }
    )
    reference = {"qubit_frequency_ghz": 4.0, "cavity_frequency_ghz": 8.0}
    requests = build_segmented_coupled_system_requests(
        row,
        resonator_type="quarter",
        run_id="unit-combined",
        reference=reference,
        sweeps=segmented_hamiltonian_sweeps(reference, qubit_count=11, bridge_count=7, resonator_count=11),
    )

    assert list(requests) == ["qubit_band", "bridge_band", "resonator_band"]
    assert requests["qubit_band"].metadata["run_id"] == "unit-combined-qubit_band"
    assert requests["bridge_band"].sweep.sweep_type == "Fast"
    assert requests["resonator_band"].setup.freq_ghz == 8.0


def test_hamiltonian_comparison_table_keeps_chi_reference_optional():
    table = hamiltonian_comparison_table(
        drivenmodal={
            "qubit_frequency_ghz": 3.9,
            "anharmonicity_mhz": -120.0,
            "cavity_frequency_ghz": 8.1,
            "kappa_mhz": 0.2,
            "g_mhz": 50.0,
            "chi_mhz": -0.04,
        },
        squadds={
            "qubit_frequency_ghz": 4.0,
            "anharmonicity_mhz": -125.0,
            "cavity_frequency_ghz": 8.0,
            "kappa_mhz": 0.25,
            "g_mhz": 48.0,
        },
    )

    assert list(table["quantity"])[-1] == "chi_mhz"
    assert math.isnan(table.iloc[-1]["squadds"])
    assert table.iloc[0]["percent_error"] == pytest.approx(-2.5)
