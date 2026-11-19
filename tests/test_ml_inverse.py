import json

import numpy as np
import pytest

from squadds.ml.inverse import MODELS, InverseModel, load_model

OUTPUTS = [
    "design_options.connection_pads.readout.claw_length",
    "design_options.connection_pads.readout.ground_spacing",
    "design_options.cross_length",
]


def tiny_params():
    """A 2 -> 2 -> 3 network whose outputs are easy to check by hand (identity scalers)."""
    return {
        "format": "squadds-mlp-v1",
        "inputs": ["qubit_frequency_GHz", "anharmonicity_MHz"],
        "outputs": OUTPUTS,
        "x_scale": [1.0, 1.0],
        "x_min": [0.0, 0.0],
        "y_scale": [1.0, 1.0, 1.0],
        "y_min": [0.0, 0.0, 0.0],
        "input_range": [[3.0, 7.0], [-500.0, -100.0]],
        "leaky_relu_slope": 0.01,
        "W1": [[1.0, 0.0], [0.0, 1.0]],
        "b1": [0.0, 0.0],
        "W2": [[1e-5, 0.0, 2e-5], [0.0, -1e-8, 0.0]],
        "b2": [0.0, 0.0, 0.0],
    }


def test_predict_units_and_columns():
    m = InverseModel(tiny_params())
    df = m.predict(5.0, -200.0)
    # h = [5, -2] after LeakyReLU; claw = 5e-5 m, ground = 2e-8 m, cross = 1e-4 m
    np.testing.assert_allclose(df[["claw_length_um", "ground_spacing_um", "cross_length_um"]].iloc[0], [50, 0.02, 100])
    assert list(df.columns[:2]) == ["qubit_frequency_GHz", "anharmonicity_MHz"] and bool(
        df["in_training_range"].iloc[0]
    )


def test_predict_many_and_out_of_range_warning():
    m = InverseModel(tiny_params())
    with pytest.warns(UserWarning, match="1 of 3 targets"):
        df = m.predict([4.0, 5.0, 9.0], -200.0)
    assert len(df) == 3 and df["in_training_range"].tolist() == [True, True, False]


def test_design_options_merges_base():
    m = InverseModel(tiny_params())
    base = {"cross_width": "30um", "connection_pads": {"readout": {"claw_width": "10um", "claw_length": "1um"}}}
    opts = m.design_options(5.0, -200.0, base=base)
    assert opts["cross_length"] == "100.000um" and opts["cross_width"] == "30um"
    assert opts["connection_pads"]["readout"] == {
        "claw_width": "10um",
        "claw_length": "50.000um",
        "ground_spacing": "0.020um",
    }
    assert base["connection_pads"]["readout"]["claw_length"] == "1um"


def test_load_model_checks_sha256(tmp_path):
    p = tmp_path / "squadds_mlp.json"
    p.write_text(json.dumps(tiny_params()))
    with pytest.raises(ValueError, match="sha256"):
        load_model(path=p)
    assert load_model(path=p, revision="local").revision == "local"
    with pytest.raises(KeyError):
        load_model("nope")


@pytest.mark.integration
def test_released_model_matches_keras():
    """Values from the original Keras model (keras 3.11.3) and its joblib scalers, in µm."""
    m = load_model()
    assert m.revision == MODELS["transmon_cross_hamiltonian_inverse"]["revision"]
    df = m.predict([4.85, 4.0, 6.5], [-205.0, -200.0, -150.0])
    keras_um = [[110.72495, 4.5716, 200.5973], [98.67366, 4.28644, 248.40778], [123.83441, 5.52171, 164.65065]]
    np.testing.assert_allclose(df[["claw_length_um", "ground_spacing_um", "cross_length_um"]], keras_um, rtol=2e-5)
