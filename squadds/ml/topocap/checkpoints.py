"""Content-addressed, atomic checkpoints for TopoCap experiments.

A checkpoint key must describe the experiment, not merely the model name.  The
fingerprint builder therefore incorporates dataset bytes, GDS bytes or an
immutable GDS manifest, split/config payloads, code identity, feature settings,
and model settings.  Loading with a different fingerprint fails by default.

Checkpoint payloads use Python pickle and must only be loaded from a trusted
local or authenticated artifact store.  A SHA-256 checksum detects accidental
corruption; it is not a substitute for artifact provenance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FINGERPRINT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


class CheckpointError(RuntimeError):
    """Base class for checkpoint validation failures."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a checkpoint belongs to a different experiment."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when checkpoint bytes or metadata fail integrity checks."""


def sha256_file(path: str | Path, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Hash one file while rejecting files that change during the read."""
    file_path = Path(path).expanduser().resolve()
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    before = file_path.stat()
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    after = file_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while it was hashed: {file_path}.")
    return digest.hexdigest()


def _canonicalize(value: Any) -> Any:
    """Convert common scientific Python values into deterministic JSON data."""
    if dataclasses.is_dataclass(value):
        return _canonicalize(dataclasses.asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonicalize(value.to_dict())
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key_string = str(key)
            if key_string in result:
                raise ValueError(f"Configuration keys collide after string conversion: {key!r}.")
            result[key_string] = _canonicalize(item)
        return result
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(canonical_items, key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fingerprint payloads cannot contain NaN or infinity.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported fingerprint value: {type(value).__name__}.")


def canonical_json(value: Any) -> str:
    """Serialize a value into stable, whitespace-free JSON."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    """Hash a canonicalized configuration or manifest payload."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scientific_runtime_identity() -> dict[str, Any]:
    """Return the numerical runtime identity that can affect fitted weights."""
    packages: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "scipy"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "<not-installed>"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }


def _logical_path(path: Path, root: Path | None) -> str:
    resolved = path.expanduser().resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.expanduser().resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"{resolved} is outside manifest root {root}.") from error
    return str(resolved)


def hash_file_manifest(
    files: Iterable[str | Path] | Mapping[str, str | Path],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic content manifest for named or positional files."""
    manifest_root = Path(root).expanduser().resolve() if root is not None else None
    if isinstance(files, Mapping):
        named_paths = [(str(name), Path(path)) for name, path in files.items()]
    else:
        paths = [Path(path) for path in files]
        named_paths = [(_logical_path(path, manifest_root), path) for path in paths]
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for logical_name, path in sorted(named_paths, key=lambda item: item[0]):
        if logical_name in seen_names:
            raise ValueError(f"Duplicate logical file name: {logical_name!r}.")
        seen_names.add(logical_name)
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Manifest entry is not a file: {resolved}.")
        digest = sha256_file(resolved)
        stat = resolved.stat()
        records.append(
            {
                "name": logical_name,
                "path": _logical_path(resolved, manifest_root),
                "size": int(stat.st_size),
                "sha256": digest,
            }
        )
    if not records:
        raise ValueError("A file manifest cannot be empty.")
    return {
        "algorithm": "sha256",
        "root": str(manifest_root) if manifest_root is not None else None,
        "files": records,
        "manifest_sha256": stable_sha256(records),
    }


def hash_gds_manifest(
    *,
    gds_files: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
    manifest: str | Path | Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Fingerprint GDS bytes or an immutable manifest that already hashes them."""
    if (gds_files is None) == (manifest is None):
        raise ValueError("Provide exactly one of gds_files or manifest.")
    if gds_files is not None:
        return {"kind": "gds-files", **hash_file_manifest(gds_files, root=root)}
    if isinstance(manifest, Mapping):
        if not manifest:
            raise ValueError("An immutable GDS manifest payload cannot be empty.")
        canonical = _canonicalize(manifest)
        return {
            "kind": "gds-manifest-payload",
            "payload": canonical,
            "manifest_sha256": stable_sha256(canonical),
        }
    manifest_path = Path(manifest).expanduser().resolve()
    return {
        "kind": "gds-manifest-file",
        **hash_file_manifest({"gds_manifest": manifest_path}, root=root),
    }


def _run_git(repository_root: Path, arguments: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def repository_code_identity(repository_root: str | Path) -> dict[str, Any]:
    """Describe a Git commit plus tracked diff and untracked source contents."""
    root = Path(repository_root).expanduser().resolve()
    try:
        commit = _run_git(root, ("rev-parse", "HEAD")).decode("ascii").strip()
        diff = _run_git(root, ("diff", "--binary", "HEAD", "--"))
        untracked_output = _run_git(root, ("ls-files", "--others", "--exclude-standard"))
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not resolve Git code identity; pass an explicit code_version instead.") from error

    source_suffixes = {
        ".cfg",
        ".ini",
        ".ipynb",
        ".json",
        ".py",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
    }
    untracked_paths = [
        root / line
        for line in untracked_output.decode("utf-8").splitlines()
        if Path(line).suffix.lower() in source_suffixes
    ]
    untracked: list[dict[str, Any]] = []
    for path in sorted(untracked_paths):
        if path.is_file():
            untracked.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    dirty_payload = {
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked,
    }
    return {
        "git_commit": commit,
        "dirty": bool(diff or untracked),
        **dirty_payload,
        "identity_sha256": stable_sha256({"commit": commit, **dirty_payload}),
    }


@dataclass(frozen=True)
class ExperimentFingerprint:
    """Verified digest and canonical scientific inputs for one experiment."""

    digest: str
    payload: Mapping[str, Any]
    schema_version: int = FINGERPRINT_SCHEMA_VERSION
    algorithm: str = "sha256"

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ValueError(f"Unsupported fingerprint algorithm: {self.algorithm!r}.")
        if self.schema_version != FINGERPRINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported fingerprint schema: {self.schema_version}.")
        canonical_payload = _canonicalize(self.payload)
        expected = stable_sha256(
            {
                "schema_version": self.schema_version,
                "algorithm": self.algorithm,
                "payload": canonical_payload,
            }
        )
        if self.digest != expected:
            raise ValueError("Experiment fingerprint digest does not match its payload.")
        object.__setattr__(self, "payload", canonical_payload)

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> ExperimentFingerprint:
        canonical_payload = _canonicalize(payload)
        digest = stable_sha256(
            {
                "schema_version": FINGERPRINT_SCHEMA_VERSION,
                "algorithm": "sha256",
                "payload": canonical_payload,
            }
        )
        return cls(digest=digest, payload=canonical_payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentFingerprint:
        return cls(
            digest=str(value["digest"]),
            payload=value["payload"],
            schema_version=int(value.get("schema_version", FINGERPRINT_SCHEMA_VERSION)),
            algorithm=str(value.get("algorithm", "sha256")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "payload": dict(self.payload),
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
        }


def build_experiment_fingerprint(
    *,
    dataset_files: Iterable[str | Path] | Mapping[str, str | Path],
    split: Any,
    feature_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    gds_files: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
    gds_manifest: str | Path | Mapping[str, Any] | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    code_version: str | Mapping[str, Any] | None = None,
    code_files: Iterable[str | Path] | Mapping[str, str | Path] | None = None,
    repository_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    gds_root: str | Path | None = None,
    require_gds: bool = True,
) -> ExperimentFingerprint:
    """Build a complete experiment fingerprint from immutable inputs.

    ``split`` may be a mapping or any object exposing ``to_dict``.  If
    ``code_version`` is omitted, ``repository_root`` is required and the Git
    commit, tracked diff, and untracked source hashes are included.  Pass
    ``code_files`` to additionally pin imported model/feature source files.
    """
    if code_version is None:
        if repository_root is None:
            raise ValueError("repository_root is required when code_version is omitted.")
        resolved_code_version: Any = repository_code_identity(repository_root)
    else:
        if (isinstance(code_version, str) and not code_version.strip()) or (
            isinstance(code_version, Mapping) and not code_version
        ):
            raise ValueError("code_version must be a non-empty immutable identifier.")
        resolved_code_version = code_version
    if hasattr(split, "to_dict") and callable(split.to_dict):
        split_payload = split.to_dict()
    elif isinstance(split, Mapping):
        split_payload = split
    else:
        raise TypeError("split must be a mapping or expose to_dict().")

    if require_gds and gds_files is None and gds_manifest is None:
        raise ValueError("A GDS file set or immutable GDS manifest is required.")
    gds_payload: Any = None
    if gds_files is not None or gds_manifest is not None:
        gds_payload = hash_gds_manifest(
            gds_files=gds_files,
            manifest=gds_manifest,
            root=gds_root,
        )

    resolved_runtime = scientific_runtime_identity() if runtime_identity is None else dict(runtime_identity)
    if not resolved_runtime:
        raise ValueError("runtime_identity must be non-empty.")
    payload: dict[str, Any] = {
        "dataset": hash_file_manifest(dataset_files, root=dataset_root),
        "gds": gds_payload,
        "split": split_payload,
        "experiment_config": dict(experiment_config or {}),
        "feature_config": dict(feature_config),
        "model_config": dict(model_config),
        "code_version": resolved_code_version,
        "runtime": resolved_runtime,
    }
    if code_files is not None:
        payload["code_files"] = hash_file_manifest(code_files, root=repository_root)
    return ExperimentFingerprint.create(payload)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A verified checkpoint payload and its provenance."""

    payload: Any
    fingerprint: ExperimentFingerprint
    metadata: Mapping[str, Any]
    path: Path
    payload_sha256: str


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _expected_digest(expected: ExperimentFingerprint | str) -> str:
    if isinstance(expected, ExperimentFingerprint):
        verified = ExperimentFingerprint.from_dict(expected.to_dict())
        return verified.digest
    digest = str(expected)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Expected checkpoint fingerprint must be a lowercase SHA-256 digest.")
    return digest


def save_checkpoint(
    path: str | Path,
    payload: Any,
    fingerprint: ExperimentFingerprint,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save a checksummed payload bound to one fingerprint.

    Existing checkpoints are never silently replaced.  A mismatched existing
    file raises :class:`CheckpointMismatchError`; a matching file requires
    ``overwrite=True``.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        fingerprint = ExperimentFingerprint.from_dict(fingerprint.to_dict())
    except (TypeError, ValueError) as error:
        raise CheckpointMismatchError("The experiment fingerprint was mutated after creation.") from error
    if checkpoint_path.exists():
        existing = load_checkpoint(checkpoint_path, expected_fingerprint=None)
        if existing.fingerprint.digest != fingerprint.digest:
            raise CheckpointMismatchError(
                "Refusing to overwrite a checkpoint from a different experiment: "
                f"{existing.fingerprint.digest} != {fingerprint.digest}."
            )
        if not overwrite:
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}.")

    payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    payload_digest = hashlib.sha256(payload_bytes).hexdigest()
    canonical_metadata = _canonicalize(dict(metadata or {}))
    header = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint.to_dict(),
        "metadata": canonical_metadata,
        "payload_sha256": payload_digest,
    }
    envelope = {
        **header,
        "envelope_sha256": stable_sha256(header),
        "payload_bytes": payload_bytes,
    }
    serialized = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(checkpoint_path, serialized)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    expected_fingerprint: ExperimentFingerprint | str | None,
) -> LoadedCheckpoint:
    """Load a trusted local checkpoint and reject corruption or mismatch."""
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        envelope = pickle.loads(checkpoint_path.read_bytes())
    except Exception as error:
        raise CheckpointCorruptionError(f"Could not decode checkpoint: {checkpoint_path}.") from error
    if not isinstance(envelope, Mapping):
        raise CheckpointCorruptionError("Checkpoint envelope is not a mapping.")
    expected_keys = {
        "schema_version",
        "fingerprint",
        "metadata",
        "payload_sha256",
        "envelope_sha256",
        "payload_bytes",
    }
    if set(envelope) != expected_keys:
        raise CheckpointCorruptionError("Checkpoint envelope fields do not match its schema.")
    if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCorruptionError(f"Unsupported checkpoint schema: {envelope.get('schema_version')!r}.")
    try:
        fingerprint = ExperimentFingerprint.from_dict(envelope["fingerprint"])
        payload_bytes = bytes(envelope["payload_bytes"])
        payload_digest = str(envelope["payload_sha256"])
        metadata = envelope.get("metadata", {})
        envelope_digest = str(envelope["envelope_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointCorruptionError("Checkpoint envelope is incomplete or invalid.") from error
    if not isinstance(metadata, Mapping):
        raise CheckpointCorruptionError("Checkpoint metadata is not a mapping.")
    try:
        canonical_metadata = _canonicalize(metadata)
        header = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint.to_dict(),
            "metadata": canonical_metadata,
            "payload_sha256": payload_digest,
        }
        actual_envelope_digest = stable_sha256(header)
    except (TypeError, ValueError) as error:
        raise CheckpointCorruptionError("Checkpoint metadata is not canonicalizable.") from error
    if actual_envelope_digest != envelope_digest:
        raise CheckpointCorruptionError("Checkpoint envelope checksum mismatch.")
    actual_digest = hashlib.sha256(payload_bytes).hexdigest()
    if actual_digest != payload_digest:
        raise CheckpointCorruptionError("Checkpoint payload checksum mismatch.")
    if expected_fingerprint is not None:
        expected_digest = _expected_digest(expected_fingerprint)
        if fingerprint.digest != expected_digest:
            raise CheckpointMismatchError(
                f"Checkpoint fingerprint mismatch: {fingerprint.digest} != {expected_digest}."
            )
    try:
        payload = pickle.loads(payload_bytes)
    except Exception as error:
        raise CheckpointCorruptionError("Checkpoint payload could not be decoded.") from error
    return LoadedCheckpoint(
        payload=payload,
        fingerprint=fingerprint,
        metadata=canonical_metadata,
        path=checkpoint_path,
        payload_sha256=payload_digest,
    )


def checkpoint_matches(path: str | Path, fingerprint: ExperimentFingerprint | str) -> bool:
    """Return whether a valid checkpoint has the expected experiment digest."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        return False
    try:
        load_checkpoint(checkpoint_path, expected_fingerprint=fingerprint)
    except (CheckpointError, OSError, ValueError):
        return False
    return True
