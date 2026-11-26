import math

import numpy as np

from squadds.simulations.drivenmodal.coupled_postprocess import (
    calculate_chi_hz,
    calculate_g_from_chi,
    calculate_kappa_hz,
    calculate_loaded_q,
    notch_half_power_linewidth,
    terminate_port_y,
    y_to_s,
)


def test_calculate_chi_hz_uses_excited_minus_ground_resonance():
    assert calculate_chi_hz(5.000e9, 5.0015e9) == 1.5e6


def test_calculate_loaded_q_and_kappa_hz_are_consistent():
    loaded_q = calculate_loaded_q(f_res_hz=6e9, fwhm_hz=2e6)
    kappa_hz = calculate_kappa_hz(f_res_hz=6e9, loaded_q=loaded_q)

    assert loaded_q == 3000
    assert math.isclose(kappa_hz, 2e6)


def test_calculate_g_from_chi_returns_positive_rate():
    value = calculate_g_from_chi(
        f_r_hz=6.0e9,
        f_q_hz=4.5e9,
        chi_hz=1.2e6,
        alpha_hz=-200e6,
    )

    assert np.isfinite(value)
    assert value > 0


def test_terminate_port_y_returns_two_port_schur_complement():
    y_matrices = np.array(
        [
            [
                [1.0 + 0.0j, 0.1 + 0.0j, 0.2 + 0.0j],
                [0.1 + 0.0j, 2.0 + 0.0j, 0.3 + 0.0j],
                [0.2 + 0.0j, 0.3 + 0.0j, 3.0 + 0.0j],
            ]
        ]
    )

    reduced = terminate_port_y(y_matrices, terminated_port=2, load_impedance_ohms=50.0)

    y_load = 1 / 50.0
    correction = 1 / (3.0 + y_load)
    expected = np.array(
        [
            [
                [1.0 - (0.2 * 0.2) * correction + 0.0j, 0.1 - (0.2 * 0.3) * correction + 0.0j],
                [0.1 - (0.3 * 0.2) * correction + 0.0j, 2.0 - (0.3 * 0.3) * correction + 0.0j],
            ]
        ]
    )
    assert np.allclose(reduced, expected)


def test_y_to_s_maps_open_circuit_to_identity_scattering():
    y_matrices = np.zeros((1, 2, 2), dtype=complex)

    s_matrices = y_to_s(y_matrices, z0_ohms=50.0)

    assert np.allclose(s_matrices[0], np.eye(2))


def test_calculate_g_from_chi_inverts_the_transmon_dispersive_relation():
    f_r, f_q, alpha, g = 8.9e9, 4.0e9, -130e6, 50e6
    delta, sigma = f_q - f_r, f_q + f_r
    chi = 2 * g**2 * (alpha / (delta * (delta + alpha)) + alpha / (sigma * (sigma + alpha)))

    g_rad_s = calculate_g_from_chi(f_r_hz=f_r, f_q_hz=f_q, chi_hz=chi, alpha_hz=alpha)

    assert math.isclose(g_rad_s / (2 * np.pi), g, rel_tol=1e-12)


def test_terminate_port_y_keeps_y_aa_for_an_ideal_short():
    y_matrices = np.array([[[2j, -0.3j], [-0.3j, 1j]]])

    reduced = terminate_port_y(y_matrices, terminated_port=1, load_impedance_ohms=0.0)

    assert np.allclose(reduced, [[[2j]]])


def test_notch_half_power_linewidth_recovers_kappa():
    f0, kappa, kappa_i = 8.9e9, 291.07e3, 20e3
    for span, points, baseline in ((5, 2_001, 1.0), (200, 40_001, None), (1700, 22_000, None)):
        freqs = np.linspace(f0 - span * kappa, f0 + span * kappa, points)
        detuning = freqs - f0
        s21 = (kappa_i + 2j * detuning) / (kappa + 2j * detuning)

        metrics = notch_half_power_linewidth(freqs, s21, baseline_power=baseline)

        assert math.isclose(metrics["fwhm_hz"], kappa, rel_tol=1e-3)
        assert math.isclose(metrics["f_res_hz"], f0, abs_tol=kappa / 1000)
