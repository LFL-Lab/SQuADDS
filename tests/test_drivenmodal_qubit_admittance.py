import math

import numpy as np

from squadds.simulations.drivenmodal.qubit_admittance import (
    combine_port_admittance_with_jj,
    extract_parallel_mode_from_total_admittance,
    extract_qubit_from_port_admittance,
    jj_parallel_admittance,
    jj_parallel_impedance,
    reduce_terminated_port_admittance,
)


def test_jj_parallel_impedance_matches_admittance_inverse():
    freqs_hz = np.linspace(3e9, 5e9, 11)
    admittance = jj_parallel_admittance(freqs_hz, lj_h=11e-9, cj_f=2e-15, rj_ohms=50_000.0)
    impedance = jj_parallel_impedance(freqs_hz, lj_h=11e-9, cj_f=2e-15, rj_ohms=50_000.0)

    np.testing.assert_allclose(admittance * impedance, np.ones_like(admittance), rtol=1e-12, atol=1e-12)


def test_extract_parallel_mode_from_total_admittance_finds_parallel_lc_zero_crossing():
    capacitance_f = 80e-15
    inductance_h = 10e-9
    freqs_hz = np.linspace(4e9, 7e9, 4001)
    omega = 2 * np.pi * freqs_hz
    y_total = 1j * omega * capacitance_f + 1 / (1j * omega * inductance_h)

    extracted = extract_parallel_mode_from_total_admittance(freqs_hz, y_total)
    expected_f_hz = 1.0 / (2 * np.pi * math.sqrt(inductance_h * capacitance_f))

    assert abs(extracted["linear_resonance_hz"] - expected_f_hz) < 1e6
    assert abs(extracted["effective_capacitance_f"] - capacitance_f) < 2e-15
    assert extracted["slope_imag_y_per_rad_s"] > 0
    assert extracted["resonance_at_sweep_edge"] is False


def test_extract_parallel_mode_from_total_admittance_marks_edge_crossing():
    capacitance_f = 80e-15
    inductance_h = 10e-9
    expected_f_hz = 1.0 / (2 * np.pi * math.sqrt(inductance_h * capacitance_f))
    freqs_hz = np.array(
        [
            expected_f_hz * 0.99,
            expected_f_hz * 1.01,
            expected_f_hz * 1.08,
            expected_f_hz * 1.15,
            expected_f_hz * 1.22,
        ]
    )
    omega = 2 * np.pi * freqs_hz
    y_total = 1j * omega * capacitance_f + 1 / (1j * omega * inductance_h)

    extracted = extract_parallel_mode_from_total_admittance(freqs_hz, y_total)

    assert extracted["resonance_at_sweep_edge"] is True


def test_extract_qubit_from_port_admittance_uses_total_capacitance_with_jj_cap():
    env_capacitance_f = 70e-15
    jj_capacitance_f = 2e-15
    inductance_h = 11e-9
    freqs_hz = np.linspace(3e9, 7e9, 8001)
    omega = 2 * np.pi * freqs_hz
    y_env = 1j * omega * env_capacitance_f

    extracted = extract_qubit_from_port_admittance(
        freqs_hz,
        y_env,
        lj_h=inductance_h,
        cj_f=jj_capacitance_f,
        rj_ohms=math.inf,
        center_hint_hz=1.0 / (2 * np.pi * math.sqrt(inductance_h * (env_capacitance_f + jj_capacitance_f))),
    )

    assert abs(extracted["effective_capacitance_f"] - (env_capacitance_f + jj_capacitance_f)) < 2e-15
    assert extracted["qubit_frequency_hz"] > 0
    assert extracted["anharmonicity_hz"] < 0
    assert extracted["ej_ghz"] > 0
    assert extracted["ec_ghz"] > 0


def test_combine_port_admittance_with_jj_adds_environment_and_junction_terms():
    freqs_hz = np.array([4e9, 5e9])
    y_env = np.array([1j * 1e-3, 1j * 2e-3], dtype=complex)

    combined = combine_port_admittance_with_jj(freqs_hz, y_env, lj_h=12e-9, cj_f=0.0, rj_ohms=math.inf)
    jj_only = jj_parallel_admittance(freqs_hz, lj_h=12e-9, cj_f=0.0, rj_ohms=math.inf)

    np.testing.assert_allclose(combined, y_env + jj_only)


def test_reduce_terminated_port_admittance_removes_short_circuit_loading_from_coupling_caps():
    c_qubit_f = 80e-15
    c_couple_f = 25e-15
    freqs_hz = np.linspace(4e9, 6e9, 5)
    omega = 2 * np.pi * freqs_hz
    y_couple = 1j * omega * c_couple_f
    y_qubit = 1j * omega * c_qubit_f

    y_matrices = np.zeros((len(freqs_hz), 3, 3), dtype=complex)
    y_matrices[:, 0, 0] = y_couple
    y_matrices[:, 1, 1] = y_couple
    y_matrices[:, 2, 2] = y_qubit + 2 * y_couple
    y_matrices[:, 0, 2] = -y_couple
    y_matrices[:, 2, 0] = -y_couple
    y_matrices[:, 1, 2] = -y_couple
    y_matrices[:, 2, 1] = -y_couple

    raw_y33 = y_matrices[:, 2, 2]
    reduced_open = reduce_terminated_port_admittance(
        y_matrices,
        target_port=2,
        terminated_port_impedances={0: np.inf, 1: np.inf},
    )

    np.testing.assert_allclose(np.imag(raw_y33), omega * (c_qubit_f + 2 * c_couple_f))
    np.testing.assert_allclose(np.imag(reduced_open), omega * c_qubit_f, rtol=1e-9, atol=1e-18)
