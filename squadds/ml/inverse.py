"""TransmonCross Hamiltonian-to-geometry model of Seidel et al., arXiv:2607.20795 (2026): a target qubit frequency and
anharmonicity in, the TransmonCross cross length, claw length and claw ground spacing out.

``load_model()`` reads ``squadds_mlp.json`` (the network's weights and scalers as numbers) from the model repository at a
pinned revision, checks its sha256 and runs the network in numpy. No extra dependencies. Same numbers as the original Keras
model and the hosted API (``squadds.ml.client``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "transmon_cross_hamiltonian_inverse": {
        "repo_id": "SQuADDS/transmon-cross-hamiltonian-inverse",
        "revision": "78180749c19b71554509b90b939a7263e25c48cd",
        "file": "squadds_mlp.json",
        "sha256": "e37af3e469de19eeb25738c9b6d4b185bbde4122cc471702a580c9058a669109",
        "paper": "https://arxiv.org/abs/2607.20795",
    }
}
DEFAULT = "transmon_cross_hamiltonian_inverse"
# model output -> (DataFrame column in µm, path in the TransmonCross options)
GEOMETRY = {
    "design_options.cross_length": ("cross_length_um", ("cross_length",)),
    "design_options.connection_pads.readout.claw_length": (
        "claw_length_um",
        ("connection_pads", "readout", "claw_length"),
    ),
    "design_options.connection_pads.readout.ground_spacing": (
        "ground_spacing_um",
        ("connection_pads", "readout", "ground_spacing"),
    ),
}


class InverseModel:
    """``predict`` gives the geometry in µm; ``design_options`` gives TransmonCross options for quantum-metal."""

    def __init__(self, params: dict, name: str = DEFAULT, revision: str | None = None):
        if params.get("format") != "squadds-mlp-v1":
            raise ValueError(f"unknown model file format {params.get('format')!r}")
        self.name, self.revision, self.params = name, revision, params
        self.inputs, self.outputs = list(params["inputs"]), list(params["outputs"])
        self.input_range = dict(zip(self.inputs, params["input_range"]))
        self._W1, self._b1 = np.asarray(params["W1"], float), np.asarray(params["b1"], float)
        self._W2, self._b2 = np.asarray(params["W2"], float), np.asarray(params["b2"], float)

    def __repr__(self) -> str:
        rng = ", ".join(f"{k} in [{lo:.4g}, {hi:.4g}]" for k, (lo, hi) in self.input_range.items())
        return f"InverseModel({self.name!r}, revision={str(self.revision)[:7]!r}; trained on {rng})"

    def _forward(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        h = (X * p["x_scale"] + p["x_min"]) @ self._W1 + self._b1
        h = np.where(h > 0, h, p["leaky_relu_slope"] * h)
        return (h @ self._W2 + self._b2 - p["y_min"]) / np.asarray(p["y_scale"])

    def predict(self, qubit_frequency_GHz, anharmonicity_MHz) -> pd.DataFrame:
        """Geometry (µm) for one target or arrays of targets. Warns for targets outside the training range."""
        f, a = np.broadcast_arrays(
            np.atleast_1d(np.asarray(qubit_frequency_GHz, float)), np.atleast_1d(np.asarray(anharmonicity_MHz, float))
        )
        X = np.column_stack([f.ravel(), a.ravel()])
        targets = dict(zip(self.inputs, X.T))
        outside = np.zeros(len(X), bool)
        for k, (lo, hi) in self.input_range.items():
            outside |= (targets[k] < lo) | (targets[k] > hi)
        if outside.any():
            rng = ", ".join(f"{k} {lo:.4g} to {hi:.4g}" for k, (lo, hi) in self.input_range.items())
            warnings.warn(
                f"{int(outside.sum())} of {len(X)} targets lie outside the training range ({rng}); "
                "check those geometries with a simulation",
                stacklevel=2,
            )
        Y = self._forward(X) * 1e6
        df = pd.DataFrame(targets)
        for j, out in enumerate(self.outputs):
            df[GEOMETRY[out][0]] = Y[:, j]
        df["in_training_range"] = ~outside
        return df

    def design_options(self, qubit_frequency_GHz: float, anharmonicity_MHz: float, base: dict | None = None) -> dict:
        """TransmonCross options for one target: ``base`` (e.g. a database row's ``design_options_qubit``) with the three
        predicted lengths put in, as strings in µm."""
        row = self.predict(qubit_frequency_GHz, anharmonicity_MHz)
        if len(row) != 1:
            raise ValueError("design_options takes one target; use predict for many")
        opts = copy.deepcopy(base) if base is not None else {}
        for col, path in GEOMETRY.values():
            d = opts
            for key in path[:-1]:
                d = d.setdefault(key, {})
            d[path[-1]] = f"{row[col].iloc[0]:.3f}um"
        return opts


def list_models() -> list[str]:
    return list(MODELS)


def load_model(
    name: str = DEFAULT, revision: str | None = None, path=None, cache_dir=None, local_files_only: bool = False
) -> InverseModel:
    """The released model, from ``path`` (a local copy of ``squadds_mlp.json``) or the Hugging Face repository at the
    pinned revision. The file must match the pinned sha256 unless another ``revision`` is given."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {list_models()}")
    entry = MODELS[name]
    rev = revision or entry["revision"]
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            entry["repo_id"], entry["file"], revision=rev, cache_dir=cache_dir, local_files_only=local_files_only
        )
    data = Path(path).read_bytes()
    if revision is None and hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{path}: sha256 differs from the pinned {entry['sha256'][:12]}; refusing to load it")
    return InverseModel(json.loads(data), name, rev)
