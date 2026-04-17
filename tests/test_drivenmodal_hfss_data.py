import numpy as np
import pandas as pd

from squadds.simulations.drivenmodal.hfss_data import (
    network_from_parameter_dataframe,
    parameter_dataframe_to_tensor,
)


def test_parameter_dataframe_to_tensor_uses_frequency_index_in_ghz():
    frame = pd.DataFrame(
        {
            "Y11": [1 + 2j, 3 + 4j],
            "Y12": [5 + 6j, 7 + 8j],
            "Y21": [9 + 10j, 11 + 12j],
            "Y22": [13 + 14j, 15 + 16j],
        },
        index=[5.0, 6.0],
    )

    freqs_hz, tensors = parameter_dataframe_to_tensor(frame, matrix_size=2, parameter_prefix="Y")

    assert np.allclose(freqs_hz, [5.0e9, 6.0e9])
    assert tensors.shape == (2, 2, 2)
    assert tensors[0, 0, 1] == 5 + 6j
    assert tensors[1, 1, 0] == 11 + 12j


def test_network_from_parameter_dataframe_builds_scikit_rf_network():
    frame = pd.DataFrame(
        {
            "S11": [0.1 + 0.0j, 0.2 + 0.0j],
            "S12": [0.3 + 0.0j, 0.4 + 0.0j],
            "S21": [0.5 + 0.0j, 0.6 + 0.0j],
            "S22": [0.7 + 0.0j, 0.8 + 0.0j],
        },
        index=[5.0, 6.0],
    )

    network = network_from_parameter_dataframe(frame, matrix_size=2, z0_ohms=50.0)

    assert np.allclose(network.f, [5.0e9, 6.0e9])
    assert network.s.shape == (2, 2, 2)
    assert np.isclose(network.s[0, 1, 0], 0.5 + 0.0j)
