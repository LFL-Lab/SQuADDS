"""Serving adapter of the planar capacitance models for the SQuADDS ML API (registry family ``planar_capacitance``).

The API Space imports ``squadds.ml.capacitance.serving:Adapter`` by its dotted path; nothing else imports this module.
``root`` is a verified local copy of the model repository (the repository layout, e.g. ``<root>/qcap/acc_v2m_sf_s0.pt``).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from . import _runtime as RT
from .models import load_model
from .predictor import CapacitancePredictor

CONVENTION = "Maxwell matrix, ground first (row and column 0), nets never merged"


class TooLarge(ValueError):
    too_large = True


class CapacitanceInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ground_layer: str = Field("1/0", description="GDS layer/datatype of the ground plane")
    conductor_layer: str = Field("1/10", description="GDS layer/datatype of the conductors")
    eps_r: float = Field(RT.TRAIN_EPS_R, gt=0, description="Substrate relative permittivity")
    thickness_um: float = Field(RT.TRAIN_THICKNESS_UM, gt=0, description="Substrate thickness in µm")
    ground_pieces: str = Field(
        "",
        description="Only when ground and conductors share a layer and the automatic ground pick is wrong: the ground's "
        "piece numbers, e.g. '1' or '1,4'",
    )
    file_base64: str | None = Field(
        None, description="The GDS file, base64 encoded (JSON requests; multipart requests send it as the field 'file')"
    )


def _layer(text: str) -> tuple[int, int]:
    try:
        layer, dtype = (int(x) for x in text.strip().split("/"))
    except ValueError as exc:
        raise ValueError(f"layer must look like '1/0', got {text!r}") from exc
    return layer, dtype


class Adapter:
    input_kind = "file"
    InputModel = CapacitanceInputs

    def __init__(self, model_id: str, root: Path, config: dict, limits: dict):
        self.model_id = model_id
        self.model = load_model(config.get("model", "QCap"), path=root)
        self.threads = int(config.get("threads", 2))
        self.predictor = CapacitancePredictor(self.model, threads=self.threads)
        self.max_conductors = int(limits.get("max_conductors", 70))

    def describe(self) -> dict:
        e = self.model.entry
        return {
            "input": "a GDS file (multipart field 'file', or inputs.file_base64 in a JSON request) and the layer and "
            "substrate fields",
            "output": "names (G first, then C1..Cn in file order), matrix_fF (N x N), sim_results (the SQuADDS_DB shape)",
            "units": "fF",
            "convention": CONVENTION,
            "model": {
                "name": self.model.name,
                "paper_name": e.get("paper_name"),
                "title": e.get("title"),
                "summary": e.get("summary"),
                "warning": e.get("warning"),
                "repo_id": self.model.repo_id,
                "revision": self.model.revision,
                "weights_sha256": self.model.weights_sha256,
            },
            "training_stack": "350 µm silicon substrate (eps_r 11.45), zero-thickness metal",
            "tested_substrates": "Si at 10-1000 µm, or eps_r 3.8-24 at 350 µm (other combinations warn)",
            "threads": self.threads,
        }

    def predict(self, inputs, file: bytes | None = None, options: dict | None = None) -> dict:
        p = CapacitanceInputs.model_validate(inputs or {})
        if not file:
            raise ValueError("no GDS file: send it as the multipart field 'file' or as inputs.file_base64")
        kwargs = {"ground_layers": (_layer(p.ground_layer),), "conductor_layers": (_layer(p.conductor_layer),)}
        if p.ground_pieces.strip():
            try:
                kwargs["ground_pieces"] = [int(x) for x in p.ground_pieces.split(",")]
            except ValueError as exc:
                raise ValueError(f"ground_pieces must look like '1' or '1,4', got {p.ground_pieces!r}") from exc
        t = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "layout.gds"
            path.write_bytes(file)
            try:
                lay = RT.prepare_layout(str(path), max_conductors=10**9, **kwargs)
            except (ValueError, OSError, RuntimeError) as exc:
                raise ValueError(f"could not read this layout: {exc}") from exc
        prepare_s = time.perf_counter() - t
        if len(lay.names) > self.max_conductors:
            raise TooLarge(
                f"{len(lay.names)} conductors found; this server takes at most {self.max_conductors} plus ground"
            )
        res = self.predictor.predict(lay, eps_r=p.eps_r, thickness_um=p.thickness_um)
        tm = res.timing_s
        meta = {
            "n_conductors": res.n_conductors,
            "checks": res.checks,
            "warnings": list(res.warnings),
            "substrate": res.substrate,
            "ground_detection": res.ground_detection,
            "threads": res.threads,
            "prior_version": res.prior_version,
            "weights_sha256": list(res.weights_sha256),
        }
        if res.learned_prior_sha256:
            meta["learned_prior_sha256"] = res.learned_prior_sha256
        return {
            "outputs": {
                "names": list(res.names),
                "matrix_fF": res.matrix_fF.tolist(),
                "units": "fF",
                "convention": CONVENTION,
                "sim_results": res.to_sim_results(),
            },
            "metadata": meta,
            "timing_s": {
                "prepare": prepare_s,
                "geometry": tm["geometry_s"],
                "prior": tm["prior_s"],
                "networks": tm["network_s"],
            },
        }

    def warm(self, example: Path) -> None:
        self.predict({}, file=Path(example).read_bytes())
