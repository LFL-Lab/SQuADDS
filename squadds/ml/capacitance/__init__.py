"""Planar capacitance models: GDS layout -> full N x N Maxwell capacitance matrix (fF, ground first).

    from squadds.ml.capacitance import CapacitancePredictor, load_model
    res = CapacitancePredictor(load_model("QCap")).predict("chip.gds")
    res.names, res.matrix_fF, res.to_sim_results()

Models: ``QCap`` (the default) and ``QCap-Fast``, both trained on all data. Physics priors: ``physics_prior`` and ``optimized_physics_prior``.
"""

from __future__ import annotations

import importlib

from .. import require_ml

_EXPORTS = {
    "CapacitancePredictor": "predictor",
    "CapacitanceResult": "result",
    "CapacitanceModel": "models",
    "ModelIntegrityError": "models",
    "list_models": "models",
    "load_model": "models",
    "PriorResult": "priors",
    "physics_prior": "priors",
    "optimized_physics_prior": "priors",
    "prepare_layout": "_runtime",
    "inspect_layout": "_runtime",
    "MLSimulator": "bridge",
    "TrainConfig": "train",
    "fit": "train",
    "fit_in_family": "train",
    "retrain_collective": "train",
    "build_shards": "training_data",
    "resolve_dataset_names": "training_data",
    "score": "benchmark",
    "PlanarCapacitanceDB": "dataset",
    "label_qc": "qc",
    "validate_capacitance_contribution": "contribute",
    "contribute_capacitance": "contribute",
    "validate_model": "contribute",
    "contribute_model": "contribute",
}
_NO_ML_NEEDED = {"PlanarCapacitanceDB", "label_qc"}


def __getattr__(name):
    if name in _EXPORTS:
        if name not in _NO_ML_NEEDED:
            require_ml()
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
