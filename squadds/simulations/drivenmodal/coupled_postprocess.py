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


def terminate_port_y(y_matrices, *, terminated_port: int, load_impedance_ohms) -> np.ndarray:
    """Terminate a single port of an N-port admittance sweep using a load impedance."""
    matrices = np.asarray(y_matrices, dtype=complex)
    if matrices.ndim != 3:
        raise ValueError("y_matrices must be a 3D array of shape (freq, nport, nport).")

    nports = matrices.shape[1]
    if matrices.shape[2] != nports:
        raise ValueError("y_matrices must be square in the last two dimensions.")
    if not 0 <= terminated_port < nports:
        raise ValueError("terminated_port is out of range.")

    load_impedance = np.asarray(load_impedance_ohms, dtype=complex)
    if load_impedance.ndim == 0:
        load_impedance = np.full(matrices.shape[0], load_impedance, dtype=complex)
    if load_impedance.shape != (matrices.shape[0],):
        raise ValueError("load_impedance_ohms must be a scalar or one value per frequency.")

    keep = [index for index in range(nports) if index != terminated_port]
    y_aa = matrices[:, keep][:, :, keep]
    y_ab = matrices[:, keep, terminated_port][:, :, np.newaxis]
    y_ba = matrices[:, terminated_port, keep][:, np.newaxis, :]
    y_bb = matrices[:, terminated_port, terminated_port]
    y_load = 1.0 / load_impedance
    denominator = (y_bb + y_load)[:, np.newaxis, np.newaxis]
    return y_aa - (y_ab * y_ba) / denominator


def y_to_s(y_matrices, *, z0_ohms: float = 50.0) -> np.ndarray:
    """Convert a sweep of admittance matrices into scattering matrices for a uniform reference impedance."""
    matrices = np.asarray(y_matrices, dtype=complex)
    if matrices.ndim != 3:
        raise ValueError("y_matrices must be a 3D array of shape (freq, nport, nport).")

    nports = matrices.shape[1]
    if matrices.shape[2] != nports:
        raise ValueError("y_matrices must be square in the last two dimensions.")

    identity = np.eye(nports, dtype=complex)
    s_matrices = np.empty_like(matrices)
    for index, y_matrix in enumerate(matrices):
        left = identity - z0_ohms * y_matrix
        right = identity + z0_ohms * y_matrix
        s_matrices[index] = left @ np.linalg.inv(right)
    return s_matrices
