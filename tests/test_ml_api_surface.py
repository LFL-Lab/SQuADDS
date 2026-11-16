"""Public surface of the planar capacitance package (snapshot): names, signatures with defaults, dataclass fields, CLI flags and
MCP tool names. A change here is a user-facing API change."""

from __future__ import annotations

import argparse
import dataclasses
import inspect

import pytest

from squadds import ml as ML

needs_ml = pytest.mark.skipif(bool(ML.missing_ml_dependencies()), reason="squadds[ml] extra not installed")

CAP_ALL = [
    "CapacitanceModel",
    "CapacitancePredictor",
    "CapacitanceResult",
    "MLSimulator",
    "ModelIntegrityError",
    "PlanarCapacitanceDB",
    "PriorResult",
    "TrainConfig",
    "build_shards",
    "contribute_capacitance",
    "contribute_model",
    "fit",
    "fit_in_family",
    "inspect_layout",
    "label_qc",
    "list_models",
    "load_model",
    "optimized_physics_prior",
    "physics_prior",
    "prepare_layout",
    "resolve_dataset_names",
    "retrain_collective",
    "score",
    "validate_capacitance_contribution",
    "validate_model",
]
SQUADDS_ALL = [
    "Analyzer",
    "SQuADDS_DB",
    "AnsysSimulator",
    "__version__",
    "__license__",
    "__copyright__",
    "__author__",
    "__status__",
    "__repo_path__",
]
SIGS = {
    "CapacitancePredictor.__init__": "(self, model: 'CapacitanceModel | str | None' = None, threads: 'int | None' = None, embed_cache_dir=None, tier=None, **load_kwargs)",
    "CapacitancePredictor.predict": "(self, layout, threads: 'int | None' = None, eps_r: 'float' = 11.45, thickness_um: 'float' = 350.0, workers: 'int | None' = None, **prepare_kwargs)",
    "load_model": "(name: 'str' = 'QCap', revision: 'str | None' = None, path=None, cache_dir=None, local_files_only: 'bool' = False, tier=None) -> 'CapacitanceModel'",
    "list_models": "() -> 'list[dict]'",
    "physics_prior": "(layout, threads: 'int | None' = None, eps_r: 'float' = 11.45, thickness_um: 'float' = 350.0, **prepare_kwargs) -> 'PriorResult'",
    "optimized_physics_prior": "(layout, threads: 'int | None' = None, eps_r: 'float' = 11.45, thickness_um: 'float' = 350.0, **prepare_kwargs) -> 'PriorResult'",
    "prepare_layout": "(layout, ground_layers=((1, 0),), conductor_layers=((1, 10),), max_conductors: 'int' = 1000, **kwargs)",
    "inspect_layout": "(layout, ground_layers=((1, 0),), conductor_layers=((1, 10),), ground_pieces=None, figsize=(6.4, 6.4), **kwargs) -> 'dict'",
    "predict_dataframe": "(df, model='QCap', component_name=None, simulator=None, **kwargs)",
    "MLSimulator": "(analyzer=None, design_options=None, model='QCap', threads: 'int | None' = None, predictor=None, clearance_um: 'float | None' = None, workdir=None, export: 'str' = 'auto', export_style: 'str | None' = None, margin_gaps: 'float | None' = None, ground_gap_um: 'float | None' = None, tier=None, **predict_kwargs)",
    "CapacitanceResult.to_dataframe": "(self)",
    "CapacitanceResult.to_dict": "(self) -> 'dict'",
    "CapacitanceResult.to_sim_results": "(self, names: 'dict | None' = None) -> 'dict'",
    "client.list_models": "(base_url: 'str | None' = None, token: 'str | None' = None, timeout: 'float' = 30) -> 'list[dict]'",
    "client.get_model": "(model_id: 'str', base_url: 'str | None' = None, token: 'str | None' = None, timeout: 'float' = 30) -> 'dict'",
    "client.predict": "(model_id: 'str', inputs: 'dict | list | None' = None, file=None, options: 'dict | None' = None, base_url: 'str | None' = None, token: 'str | None' = None, timeout: 'float' = 600) -> 'dict'",
    "Adapter.__init__": "(self, model_id: 'str', root: 'Path', config: 'dict', limits: 'dict')",
    "Adapter.describe": "(self) -> 'dict'",
    "Adapter.predict": "(self, inputs, file: 'bytes | None' = None, options: 'dict | None' = None) -> 'dict'",
    "Adapter.warm": "(self, example: 'Path') -> 'None'",
}
FIELDS = {
    "CapacitanceResult": [
        "names",
        "matrix_fF",
        "model",
        "title",
        "prior_version",
        "prior_stack",
        "weights_sha256",
        "learned_prior_sha256",
        "timing_s",
        "threads",
        "machine",
        "checks",
        "warnings",
        "substrate",
        "ground_detection",
        "source",
        "layout_hash",
        "cached",
        "package",
        "units",
        "convention",
    ],
    "PriorResult": [
        "names",
        "C_fF",
        "solves_fF",
        "levels_ppg",
        "kind",
        "prior_version",
        "stack",
        "alpha",
        "timing_s",
        "threads",
        "machine",
        "layout_hash",
        "units",
    ],
    "CapacitanceModel": ["name", "entry", "root", "repo_id", "revision", "sha256"],
}
INPUTS = ["ground_layer", "conductor_layer", "eps_r", "thickness_um", "ground_pieces", "file_base64"]
CLI_PREDICT = {
    "gds": None,
    "--model": "QCap",
    "--model-path": None,
    "--revision": None,
    "--threads": None,
    "--workers": 1,
    "--ground": "1/0",
    "--conductor": "1/10",
    "--ground-pieces": None,
    "--eps-r": 11.45,
    "--thickness-um": 350.0,
    "--format": "csv",
}
MCP_CAPML = [
    "list_capacitance_models",
    "predict_capacitance_from_gds",
    "query_planar_capacitance",
    "get_planar_capacitance_conventions",
    "validate_capacitance_contribution",
    "compute_capacitance_prior",
    "inspect_capacitance_layout",
    "get_planar_capacitance_design",
    "nearest_planar_capacitance_designs",
]
MCP_HOSTED = ["list_hosted_ml_models", "predict_with_hosted_ml_model"]


def test_top_level_names():
    import squadds

    assert squadds.__all__ == SQUADDS_ALL
    from squadds import Analyzer, SQuADDS_DB  # noqa: F401

    assert inspect.isclass(squadds.Analyzer) and inspect.isclass(squadds.SQuADDS_DB)
    import squadds.ml.capacitance as cap

    assert cap.__all__ == CAP_ALL


def test_top_level_ansys_simulator_resolves():
    import squadds

    try:
        cls = squadds.AnsysSimulator
    except ImportError as e:  # Qt / qiskit-metal need a display on some machines
        pytest.skip(f"AnsysSimulator needs a GUI stack here: {e}")
    assert inspect.isclass(cls)


@needs_ml
def test_signatures_and_fields():
    import squadds.ml.capacitance as cap
    from squadds.ml import client
    from squadds.ml.capacitance import serving
    from squadds.ml.capacitance.bridge import predict_dataframe

    R = cap.CapacitanceResult
    funcs = {
        "CapacitancePredictor.__init__": cap.CapacitancePredictor.__init__,
        "CapacitancePredictor.predict": cap.CapacitancePredictor.predict,
        "load_model": cap.load_model,
        "list_models": cap.list_models,
        "physics_prior": cap.physics_prior,
        "optimized_physics_prior": cap.optimized_physics_prior,
        "prepare_layout": cap.prepare_layout,
        "inspect_layout": cap.inspect_layout,
        "predict_dataframe": predict_dataframe,
        "MLSimulator": cap.MLSimulator,
        "CapacitanceResult.to_dataframe": R.to_dataframe,
        "CapacitanceResult.to_dict": R.to_dict,
        "CapacitanceResult.to_sim_results": R.to_sim_results,
        "client.list_models": client.list_models,
        "client.get_model": client.get_model,
        "client.predict": client.predict,
        "Adapter.__init__": serving.Adapter.__init__,
        "Adapter.describe": serving.Adapter.describe,
        "Adapter.predict": serving.Adapter.predict,
        "Adapter.warm": serving.Adapter.warm,
    }
    assert {k: str(inspect.signature(f)) for k, f in funcs.items()} == SIGS
    assert isinstance(inspect.getattr_static(R, "n_conductors"), property)
    assert {
        c.__name__: [f.name for f in dataclasses.fields(c)] for c in (R, cap.PriorResult, cap.CapacitanceModel)
    } == FIELDS
    assert list(serving.CapacitanceInputs.model_fields) == INPUTS
    assert client.DEFAULT_URL == "https://squadds-squadds-ml-inference-api.hf.space"


def test_cli_predict_flags(monkeypatch):
    from squadds.ml.capacitance import cli

    seen = {}

    def grab(self, *a, **k):
        seen["parser"] = self
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(SystemExit):
        cli.main(["predict", "x.gds"])
    sub = next(a for a in seen["parser"]._actions if isinstance(a, argparse._SubParsersAction))
    assert sorted(sub.choices) == ["predict", "train"]
    acts = [a for a in sub.choices["predict"]._actions if a.dest != "help"]
    assert {(a.option_strings[0] if a.option_strings else a.dest): a.default for a in acts} == CLI_PREDICT


def test_mcp_tool_names():
    pytest.importorskip("mcp")
    from squadds_mcp.tools import capacitance_ml, hosted_ml

    class Fake:
        def __init__(self):
            self.names = []

        def tool(self, *a, **k):
            return lambda f: (self.names.append(f.__name__), f)[1]

    f = Fake()
    capacitance_ml.register_capacitance_ml_tools(f)
    assert f.names == MCP_CAPML
    f = Fake()
    hosted_ml.register_hosted_ml_tools(f)
    assert f.names == MCP_HOSTED
