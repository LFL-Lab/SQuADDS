"""Helpers for extracting qubit properties from driven-modal port admittance."""

from __future__ import annotations

import math

import numpy as np
import scqubits as scq
from scipy import constants


def bare_lj_to_ej_ghz(lj_h: float) -> float:
    """Convert a bare Josephson inductance into EJ in GHz for scqubits."""
    if lj_h <= 0:
        raise ValueError("lj_h must be positive.")
    phi0_over_2pi = constants.hbar / (2 * constants.e)
    ej_joules = (phi0_over_2pi**2) / lj_h
    return float(ej_joules / constants.h / 1e9)


def capacitance_to_ec_ghz(c_total_f: float) -> float:
    """Convert a total shunting capacitance in Farads into EC in GHz for scqubits."""
    if c_total_f <= 0:
        raise ValueError("c_total_f must be positive.")
    ec_joules = constants.e**2 / (2 * c_total_f)
    return float(ec_joules / constants.h / 1e9)


def jj_parallel_admittance(
    freqs_hz: np.ndarray,
    *,
    lj_h: float,
    cj_f: float = 0.0,
    rj_ohms: float = math.inf,
) -> np.ndarray:
    """Return the parallel RLC admittance of a Josephson junction surrogate."""
    frequencies = np.asarray(freqs_hz, dtype=float)
    if frequencies.ndim != 1:
        raise ValueError("freqs_hz must be a 1D array.")
    if np.any(frequencies <= 0):
        raise ValueError("freqs_hz must be strictly positive.")
    if lj_h <= 0:
        raise ValueError("lj_h must be positive.")
    if cj_f < 0:
        raise ValueError("cj_f cannot be negative.")
    if rj_ohms <= 0:
        raise ValueError("rj_ohms must be positive or infinity.")

    omega = 2 * np.pi * frequencies
    admittance = np.zeros_like(frequencies, dtype=complex)
    if np.isfinite(rj_ohms):
        admittance += 1.0 / rj_ohms
    if cj_f > 0:
        admittance += 1j * omega * cj_f
    admittance += 1.0 / (1j * omega * lj_h)
    return admittance


def jj_parallel_impedance(
    freqs_hz: np.ndarray,
    *,
    lj_h: float,
    cj_f: float = 0.0,
    rj_ohms: float = math.inf,
) -> np.ndarray:
    """Return the equivalent impedance of the parallel JJ RLC model."""
    admittance = jj_parallel_admittance(freqs_hz, lj_h=lj_h, cj_f=cj_f, rj_ohms=rj_ohms)
    impedance = np.full_like(admittance, np.inf, dtype=complex)
    nonzero = np.abs(admittance) > 0
    impedance[nonzero] = 1.0 / admittance[nonzero]
    return impedance


def combine_port_admittance_with_jj(
    freqs_hz: np.ndarray,
    y33_env: np.ndarray,
    *,
    lj_h: float,
    cj_f: float = 0.0,
    rj_ohms: float = math.inf,
) -> np.ndarray:
    """Return the total small-signal admittance seen by the JJ port."""
    y_env = np.asarray(y33_env, dtype=complex)
    if y_env.ndim != 1:
        raise ValueError("y33_env must be a 1D array.")
    if y_env.shape != np.asarray(freqs_hz).shape:
        raise ValueError("freqs_hz and y33_env must have the same shape.")
    return y_env + jj_parallel_admittance(freqs_hz, lj_h=lj_h, cj_f=cj_f, rj_ohms=rj_ohms)


def _interpolate_zero_crossing(
    omega_left: float,
    omega_right: float,
    imag_left: float,
    imag_right: float,
) -> float:
    if imag_right == imag_left:
        return float((omega_left + omega_right) / 2)
    return float(omega_left - imag_left * (omega_right - omega_left) / (imag_right - imag_left))


def _fit_imaginary_slope(
    omega: np.ndarray,
    imag_y: np.ndarray,
    center_index: int,
    *,
    half_window: int = 2,
) -> float:
    start = max(0, center_index - half_window)
    stop = min(len(omega), center_index + half_window + 2)
    local_omega = omega[start:stop]
    local_imag = imag_y[start:stop]
    if len(local_omega) < 2:
        raise ValueError("At least two points are required to estimate the admittance slope.")
    degree = 1 if len(local_omega) < 4 else 2
    coeffs = np.polyfit(local_omega, local_imag, degree)
    if degree == 1:
        return float(coeffs[0])
    a, b, _ = coeffs
    omega_center = omega[center_index]
    return float(2 * a * omega_center + b)


def extract_parallel_mode_from_total_admittance(
    freqs_hz: np.ndarray,
    y_total: np.ndarray,
    *,
    center_hint_hz: float | None = None,
) -> dict[str, float]:
    """Extract the linearized qubit mode from a total JJ-port admittance trace.

    The resonance is identified from zero crossings of ``Im[Y_total]`` with a
    positive slope, which correspond to parallel resonances of the full linear
    environment plus the JJ surrogate. The effective capacitance follows from
    ``0.5 * d(Im[Y]) / dω`` at the zero crossing.
    """
    frequencies = np.asarray(freqs_hz, dtype=float)
    y_trace = np.asarray(y_total, dtype=complex)
    if frequencies.ndim != 1 or y_trace.ndim != 1 or frequencies.shape != y_trace.shape:
        raise ValueError("freqs_hz and y_total must be matching 1D arrays.")
    if len(frequencies) < 3:
        raise ValueError("At least three frequency points are required.")

    omega = 2 * np.pi * frequencies
    imag_y = np.imag(y_trace)
    real_y = np.real(y_trace)
    sign_change_indices = np.where(np.signbit(imag_y[:-1]) != np.signbit(imag_y[1:]))[0]
    candidates: list[dict[str, float]] = []
    for index in sign_change_indices:
        slope_pair = (imag_y[index + 1] - imag_y[index]) / (omega[index + 1] - omega[index])
        if slope_pair <= 0:
            continue
        omega_zero = _interpolate_zero_crossing(
            omega[index],
            omega[index + 1],
            imag_y[index],
            imag_y[index + 1],
        )
        freq_zero_hz = omega_zero / (2 * np.pi)
        slope = _fit_imaginary_slope(omega, imag_y, index)
        effective_capacitance_f = 0.5 * slope
        if effective_capacitance_f <= 0:
            continue
        real_zero = np.interp(omega_zero, omega, real_y)
        candidates.append(
            {
                "linear_resonance_hz": float(freq_zero_hz),
                "effective_capacitance_f": float(effective_capacitance_f),
                "slope_imag_y_per_rad_s": float(slope),
                "real_admittance_s": float(real_zero),
                "crossing_index": float(index),
            }
        )

    if not candidates:
        raise ValueError("No positive-slope zero crossing was found in Im[Y].")

    if center_hint_hz is None:
        selected = min(candidates, key=lambda candidate: abs(candidate["real_admittance_s"]))
    else:
        selected = min(candidates, key=lambda candidate: abs(candidate["linear_resonance_hz"] - center_hint_hz))
    return selected


def extract_qubit_from_port_admittance(
    freqs_hz: np.ndarray,
    y33_env: np.ndarray,
    *,
    lj_h: float,
    cj_f: float = 0.0,
    rj_ohms: float = math.inf,
    center_hint_hz: float | None = None,
    ncut: int = 35,
) -> dict[str, float]:
    """Estimate transmon parameters from the JJ-port admittance and a JJ RLC model."""
    scq.set_units("GHz")

    y_total = combine_port_admittance_with_jj(
        freqs_hz,
        y33_env,
        lj_h=lj_h,
        cj_f=cj_f,
        rj_ohms=rj_ohms,
    )
    mode = extract_parallel_mode_from_total_admittance(
        freqs_hz,
        y_total,
        center_hint_hz=center_hint_hz,
    )
    ej_ghz = bare_lj_to_ej_ghz(lj_h)
    ec_ghz = capacitance_to_ec_ghz(mode["effective_capacitance_f"])
    transmon = scq.Transmon(EJ=ej_ghz, EC=ec_ghz, ng=0, ncut=ncut)
    mode.update(
        {
            "ej_ghz": float(ej_ghz),
            "ec_ghz": float(ec_ghz),
            "qubit_frequency_hz": float(transmon.E01() * 1e9),
            "anharmonicity_hz": float(transmon.anharmonicity() * 1e9),
            "jj_capacitance_f": float(cj_f),
            "jj_resistance_ohms": float(rj_ohms) if np.isfinite(rj_ohms) else math.inf,
            "lj_h": float(lj_h),
        }
    )
    return mode


__all__ = [
    "bare_lj_to_ej_ghz",
    "capacitance_to_ec_ghz",
    "combine_port_admittance_with_jj",
    "extract_parallel_mode_from_total_admittance",
    "extract_qubit_from_port_admittance",
    "jj_parallel_admittance",
    "jj_parallel_impedance",
]
