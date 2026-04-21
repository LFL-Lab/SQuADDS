"""Capacitance extraction helpers for frequency-dependent driven-modal data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def capacitance_matrix_from_y(freq_hz: float, y_matrix) -> np.ndarray:
    """Compute the capacitance matrix from the imaginary part of the admittance matrix."""
    if freq_hz <= 0:
        raise ValueError("freq_hz must be positive.")
    omega = 2 * np.pi * freq_hz
    c_matrix = np.imag(np.asarray(y_matrix, dtype=complex)) / omega
    return 0.5 * (c_matrix + c_matrix.T)


def maxwell_capacitance_dataframe(
    c_matrix,
    *,
    node_names: list[str],
    ground_name: str = "ground",
) -> pd.DataFrame:
    """Expand an active-node capacitance matrix into a full Maxwell matrix with ground."""
    active = np.asarray(c_matrix, dtype=float)
    if active.shape != (len(node_names), len(node_names)):
        raise ValueError("node_names must match the capacitance matrix dimensions.")

    size = active.shape[0]
    maxwell = np.zeros((size + 1, size + 1), dtype=float)
    maxwell[:size, :size] = active
    row_sums = active.sum(axis=1)
    maxwell[:size, size] = -row_sums
    maxwell[size, :size] = -row_sums
    maxwell[size, size] = row_sums.sum()
    labels = [*node_names, ground_name]
    return pd.DataFrame(maxwell, index=labels, columns=labels)


def capacitance_dataframe_from_y_sweep(freqs_hz, y_matrices, node_names: list[str]) -> pd.DataFrame:
    """Flatten a Y-parameter sweep into a dataframe of capacitance entries by node pair."""
    freqs = np.asarray(freqs_hz, dtype=float)
    matrices = np.asarray(y_matrices, dtype=complex)
    if matrices.shape[0] != len(freqs):
        raise ValueError("y_matrices must have the same leading dimension as freqs_hz.")
    if matrices.shape[1] != len(node_names) or matrices.shape[2] != len(node_names):
        raise ValueError("node_names must match the Y-matrix dimensions.")

    rows = []
    for freq_hz, y_matrix in zip(freqs, matrices, strict=True):
        c_matrix = capacitance_matrix_from_y(freq_hz, y_matrix)
        row = {"frequency_hz": freq_hz}
        for row_name, row_values in zip(node_names, c_matrix, strict=True):
            for column_name, value in zip(node_names, row_values, strict=True):
                row[f"{row_name}__{column_name}_F"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)
