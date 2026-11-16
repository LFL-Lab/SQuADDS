"""Released planar-capacitance models: manifest, download (Hugging Face, pinned revision) and sha256 verification.

Sources, in order: ``path=`` (a local bundle: the repository layout, e.g. ``<path>/qcap/acc_v2m_sf_s0.pt``), the
environment variable ``SQUADDS_CAPML_MODELS`` (a directory holding one bundle per repository, named after it, e.g.
``$SQUADDS_CAPML_MODELS/QCap``), else ``huggingface_hub.hf_hub_download`` from the manifest's repository
at its pinned revision. Every file is checked against the sha256 in the package manifest; a mismatch is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ._runtime import TRAIN_EPS_R

ENV_MODELS = "SQUADDS_CAPML_MODELS"
MANIFEST = json.loads((Path(__file__).with_name("manifest.json")).read_text(encoding="utf-8"))
DISPLAY_NAMES = {"qcap": "QCap", "qcap-fast": "QCap-Fast"}
_RETIRED = {"flagship": "QCap", "accurate": "QCap", "fast": "QCap-Fast"}
_SHA_MEMO: dict = {}


class ModelIntegrityError(RuntimeError):
    """A model file is missing or its sha256 differs from the package manifest."""


class _Registry:
    """The registry view the frozen stage (``_core.paper``) reads: one model as a ``paper_arms`` entry."""

    def __init__(self, data: dict, root: Path, key: str):
        self.data, self.root, self.path = data, Path(root), key

    def __getitem__(self, k):
        return self.data[k]

    def eps_of(self, substrate) -> float:
        if isinstance(substrate, (int, float)):
            return float(substrate)
        if str(substrate).lower().startswith("si"):
            return TRAIN_EPS_R
        return float(substrate)


def registry_view(arms: dict, root, key: str) -> _Registry:
    data = {
        "version": MANIFEST["registry_version"],
        "paper_arms": arms,
        "paper_hub": {"prior_stacks": MANIFEST["prior_stacks"], "default_arm": next(iter(arms), None)},
    }
    return _Registry(data, root, key)


def sha256_file(p) -> str:
    p = Path(p)
    st = p.stat()
    k = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if k not in _SHA_MEMO:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _SHA_MEMO[k] = h.hexdigest()
    return _SHA_MEMO[k]


def model_files(entry: dict) -> list[tuple[str, str]]:
    """(relative path, sha256) of every file a model needs."""
    out = [(c["path"], c["sha256"]) for c in entry.get("checkpoints") or []]
    if entry.get("learned_prior"):
        c = entry["learned_prior"]["checkpoint"]
        out.append((c["path"], c["sha256"]))
    return out


def list_models() -> list[dict]:
    """Served models (``QCap``, ``QCap-Fast``) with their numbers (median % with MAE / RMSE in fF) and warnings.
    ``reference`` = the held-out numbers of the same recipe with networks trained on the TRAIN split only (reported in the
    paper), not of the served model; ``own`` = the served model's in-sample fit; ``b5`` = its one never-seen check."""
    return [
        {
            "model": DISPLAY_NAMES[name],
            "key": name,
            "default": name == MANIFEST["default_model"],
            "title": e.get("title"),
            "repo_id": e["repo_id"],
            "revision": e["revision"],
            "warning": e.get("warning"),
            "reference": e.get("reference") or {},
            "own": e.get("own") or {},
            "b5": e.get("b5"),
            "time_x86": e.get("time_x86"),
        }
        for name, e in MANIFEST["models"].items()
    ]


@dataclass
class CapacitanceModel:
    """A verified local copy of one model."""

    name: str
    entry: dict
    root: Path
    repo_id: str
    revision: str | None
    sha256: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """The manifest key (``qcap``, ``qcap-fast``); ``name`` is the display name (``QCap``, ``QCap-Fast``)."""
        return {v: k for k, v in DISPLAY_NAMES.items()}.get(self.name, self.name)

    @property
    def title(self) -> str:
        return self.entry.get("title", self.name)

    @property
    def warning(self) -> str | None:
        return self.entry.get("warning")

    @property
    def weights_sha256(self) -> list[str]:
        return [c["sha256"] for c in self.entry["checkpoints"]]

    @property
    def learned_prior_sha256(self) -> str | None:
        lp = self.entry.get("learned_prior")
        return lp["checkpoint"]["sha256"] if lp else None

    def registry(self) -> _Registry:
        return registry_view({self.name: self.entry}, self.root, f"{self.repo_id}@{self.revision}:{self.root}")

    def __repr__(self) -> str:
        return f"CapacitanceModel({self.name!r}, root={str(self.root)!r})"


def model_key(name: str) -> str:
    """Manifest key of a released model name: case-insensitive, ``-`` and ``_`` interchangeable (``QCap-Fast``, ``qcap_fast``).
    The pre-release names ``flagship`` / ``fast`` raise a ValueError naming the new ones; other names a KeyError."""
    k = str(name).strip().lower().replace("_", "-")
    if k in MANIFEST["models"]:
        return k
    if k in _RETIRED:
        raise ValueError(
            f"model {name!r} was renamed: use load_model({_RETIRED[k]!r}) (models: {', '.join(DISPLAY_NAMES.values())})"
        )
    raise KeyError(f"unknown model {name!r}; available: {sorted(DISPLAY_NAMES.values())}")


def is_model_name(name) -> bool:
    """True when ``name`` is a released model's name (see ``model_key``; a retired name raises its ValueError)."""
    if not isinstance(name, str):
        return False
    try:
        model_key(name)
    except KeyError:
        return False
    return True


def no_tier(tier) -> None:
    """Models are named directly since tiers were removed: refuse ``tier=`` with the call to use instead."""
    if tier is not None:
        new = "QCap-Fast" if str(tier).lower() == "fast" else "QCap"
        names = ", ".join(DISPLAY_NAMES.values())
        raise TypeError(f"tier= was removed: the models are named directly, e.g. load_model({new!r}) (models: {names})")


def verify(root, entry: dict) -> list[str]:
    """Check every file of a model under root against the manifest sha256 (ModelIntegrityError otherwise)."""
    out = []
    for rel, h in model_files(entry):
        p = Path(root) / rel
        if not p.exists():
            raise ModelIntegrityError(f"model file missing: {p}")
        got = sha256_file(p)
        if got != h:
            raise ModelIntegrityError(
                f"{p}: sha256 {got[:12]} differs from the manifest ({h[:12]}); refusing to load it"
            )
        out.append(got)
    return out


def _download(entry: dict, revision: str, cache_dir=None, local_files_only: bool = False) -> Path:
    from huggingface_hub import hf_hub_download

    root = None
    for rel, _ in model_files(entry):
        p = Path(
            hf_hub_download(
                repo_id=entry["repo_id"],
                filename=rel,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
        r = p
        for _ in Path(rel).parts:
            r = r.parent
        if root is not None and r != root:
            raise ModelIntegrityError(f"files of {entry['repo_id']} resolved to different snapshots ({root}, {r})")
        root = r
    return root


def load_model(
    name: str = "QCap",
    revision: str | None = None,
    path=None,
    cache_dir=None,
    local_files_only: bool = False,
    tier=None,
) -> CapacitanceModel:
    """A released model: ``load_model("QCap")`` (the default) or ``load_model("QCap-Fast")``. Names are case-insensitive,
    with ``-`` and ``_`` interchangeable."""
    no_tier(tier)
    key = model_key(name)
    entry = MANIFEST["models"][key]
    rev = revision or entry["revision"]
    repo_dir = entry["repo_id"].split("/")[-1]
    if path is not None:
        root = Path(path)
    elif os.environ.get(ENV_MODELS):
        base = Path(os.environ[ENV_MODELS])
        root = base / repo_dir if (base / repo_dir).exists() else base
    else:
        root = _download(entry, rev, cache_dir, local_files_only)
    sha = verify(root, entry)
    return CapacitanceModel(DISPLAY_NAMES[key], entry, root, entry["repo_id"], rev, sha)
