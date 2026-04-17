"""Helpers for parsing HFSS parameter tables into array and network structures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skrf as rf


def parameter_dataframe_to_tensor(
    frame: pd.DataFrame,
    *,
    matrix_size: int,
    parameter_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Qiskit Metal HFSS parameter dataframe into a frequency sweep tensor."""
    if matrix_size < 1:
        raise ValueError("matrix_size must be at least 1.")

    freqs_hz = frame.index.to_numpy(dtype=float) * 1e9
    matrices = np.zeros((len(frame), matrix_size, matrix_size), dtype=complex)

    for measure in range(matrix_size):
        for excite in range(matrix_size):
            column_name = f"{parameter_prefix}{measure + 1}{excite + 1}"
            if column_name not in frame.columns:
                raise ValueError(f"Missing parameter column: {column_name}")
            matrices[:, measure, excite] = frame[column_name].to_numpy(dtype=complex)

    return freqs_hz, matrices


def network_from_parameter_dataframe(
    frame: pd.DataFrame,
    *,
    matrix_size: int,
    z0_ohms: float = 50.0,
) -> rf.Network:
    """Build a scikit-rf Network from an HFSS S-parameter dataframe."""
    freqs_hz, s_matrices = parameter_dataframe_to_tensor(
        frame,
        matrix_size=matrix_size,
        parameter_prefix="S",
    )
    frequency = rf.Frequency.from_f(freqs_hz, unit="hz")
    return rf.Network(frequency=frequency, s=s_matrices, z0=z0_ohms)


def write_touchstone_from_dataframe(
    frame: pd.DataFrame,
    *,
    matrix_size: int,
    output_path: str | Path,
    z0_ohms: float = 50.0,
) -> Path:
    """Write a Touchstone file from an HFSS S-parameter dataframe."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    network = network_from_parameter_dataframe(frame, matrix_size=matrix_size, z0_ohms=z0_ohms)
    filename = path.name
    if filename.endswith(".s2p") or filename.endswith(".s3p") or filename.endswith(".s4p"):
        network.write_touchstone(filename=path.with_suffix("").as_posix())
    else:
        network.write_touchstone(filename=path.as_posix())
    return path
