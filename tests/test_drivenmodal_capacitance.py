import numpy as np

from squadds.simulations.drivenmodal.capacitance import (
    capacitance_dataframe_from_y_sweep,
    capacitance_matrix_from_y,
    maxwell_capacitance_dataframe,
)


def test_capacitance_matrix_from_y_divides_imaginary_part_by_omega():
    freq_hz = 5e9
    y_matrix = 1j * 2 * np.pi * freq_hz * np.array([[10e-15, -2e-15], [-2e-15, 9e-15]])

    result = capacitance_matrix_from_y(freq_hz, y_matrix)

    assert np.allclose(result, [[10e-15, -2e-15], [-2e-15, 9e-15]])


def test_capacitance_dataframe_from_y_sweep_flattens_entries_by_node_name():
    freqs_hz = np.array([5e9, 6e9])
    y_matrices = np.array(
        [
            1j * 2 * np.pi * freqs_hz[0] * np.array([[10e-15, -2e-15], [-2e-15, 9e-15]]),
            1j * 2 * np.pi * freqs_hz[1] * np.array([[11e-15, -3e-15], [-3e-15, 8e-15]]),
        ]
    )

    frame = capacitance_dataframe_from_y_sweep(freqs_hz, y_matrices, node_names=["top", "bottom"])

    assert list(frame.columns) == ["frequency_hz", "top__top_F", "top__bottom_F", "bottom__top_F", "bottom__bottom_F"]
    assert np.isclose(frame.iloc[0]["top__top_F"], 10e-15)
    assert np.isclose(frame.iloc[1]["bottom__bottom_F"], 8e-15)


def test_maxwell_capacitance_dataframe_adds_ground_row_and_column():
    frame = maxwell_capacitance_dataframe(
        np.array([[10e-15, -2e-15], [-2e-15, 9e-15]]),
        node_names=["cross", "claw"],
    )

    assert list(frame.index) == ["cross", "claw", "ground"]
    assert np.isclose(frame.loc["cross", "ground"], -8e-15)
    assert np.isclose(frame.loc["ground", "ground"], 15e-15)
