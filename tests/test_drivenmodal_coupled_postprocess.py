import math

import numpy as np

from squadds.simulations.drivenmodal.coupled_postprocess import (
    calculate_chi_hz,
    calculate_g_from_chi,
    calculate_kappa_hz,
    calculate_loaded_q,
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
