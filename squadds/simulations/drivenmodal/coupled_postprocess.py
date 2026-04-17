"""Pure helpers for driven-modal coupled-system post-processing."""

from __future__ import annotations

import math

import numpy as np


def calculate_chi_hz(f_ground_hz: float, f_excited_hz: float) -> float:
    """Return the dispersive shift in Hz from the embedded ground/excited resonances."""
    return float(f_excited_hz - f_ground_hz)


def calculate_loaded_q(*, f_res_hz: float, fwhm_hz: float) -> float:
    """Compute loaded quality factor from resonance frequency and FWHM."""
    if f_res_hz <= 0:
        raise ValueError("f_res_hz must be positive.")
    if fwhm_hz <= 0:
        raise ValueError("fwhm_hz must be positive.")
    return float(f_res_hz / fwhm_hz)


def calculate_kappa_hz(*, f_res_hz: float, loaded_q: float) -> float:
    """Compute kappa / 2pi in Hz from loaded quality factor."""
    if f_res_hz <= 0:
        raise ValueError("f_res_hz must be positive.")
    if loaded_q <= 0:
        raise ValueError("loaded_q must be positive.")
    return float(f_res_hz / loaded_q)


def calculate_g_from_chi(*, f_r_hz: float, f_q_hz: float, chi_hz: float, alpha_hz: float) -> float:
    """Back-calculate coupling strength magnitude using the dispersive transmon relation with the non-RWA term."""
    omega_r = 2 * np.pi * f_r_hz
    omega_q = 2 * np.pi * f_q_hz
    chi = 2 * np.pi * chi_hz
    alpha = 2 * np.pi * alpha_hz
    delta = omega_q - omega_r
    sigma = omega_q + omega_r

    rwa_term = alpha / (delta * (delta - alpha))
    non_rwa_term = alpha / (sigma * (sigma + alpha))
    denominator = 2 * (rwa_term + non_rwa_term)
    if denominator == 0:
        raise ValueError("The dispersive denominator is zero; cannot compute coupling.")

    # ``chi`` sign conventions vary across workflows. We expose a positive coupling
    # magnitude while preserving the user's chosen resonance ordering elsewhere.
    g_squared_magnitude = abs(chi / denominator)
    return float(math.sqrt(g_squared_magnitude))
