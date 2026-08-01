"""Build a compact, checksummed publication bundle from a TopoCap study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

REPORT_ARTIFACTS = (
    "data_audit",
    "learning_curves",
    "uncertainty",
    "ablations",
    "topology_checks",
    "diffusion_decision",
)
PUBLISHABLE_STATUSES = {"COMPLETE", "EXPLORATORY_COMPLETE"}
STUDY_STATE_NAME = "study-state.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a complete TopoCap result directory and publish only the compact report "
            "tables plus a self-describing manifest."
        )
    )
    parser.add_argument("source_dir", type=Path, help="Complete TopoCap study directory.")
    parser.add_argument("destination_dir", type=Path, help="Compact publication directory.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"Artifact path escapes source directory: {relative_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _portable_cache_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Cache identity must be an object")
    return {key: item for key, item in value.items() if key not in {"path", "manifest_path"}}


def package_results(source_dir: Path, destination_dir: Path) -> Path:
    """Verify and copy the compact report contract into ``destination_dir``."""

    source_dir = source_dir.expanduser().resolve()
    destination_dir = destination_dir.expanduser().resolve()
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("study_status")
    if status not in PUBLISHABLE_STATUSES:
        raise ValueError(f"Study status is not publishable: {status!r}")

    state_path = source_dir / STUDY_STATE_NAME
    study_state = json.loads(state_path.read_text())
    if not isinstance(study_state, dict) or study_state.get("digest") != manifest.get("state_digest"):
        raise ValueError("Study-state digest does not match the result manifest")
    configuration = study_state.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Study state has no configuration object")
    configuration["cache"] = _portable_cache_identity(configuration.get("cache"))
    destination_state = destination_dir / STUDY_STATE_NAME
    _atomic_json(destination_state, study_state)
    state_descriptor = {
        "path": destination_state.name,
        "sha256": _sha256(destination_state),
        "schema_version": "topocap-study-state-v1",
    }

    source_descriptors = manifest.get("artifacts")
    if not isinstance(source_descriptors, dict):
        raise ValueError("Study manifest has no artifact descriptor mapping")

    compact_descriptors: dict[str, Any] = {}
    for logical_name in REPORT_ARTIFACTS:
        descriptor = source_descriptors.get(logical_name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"Missing descriptor for {logical_name!r}")
        relative_path = descriptor.get("path")
        expected_hash = descriptor.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Incomplete descriptor for {logical_name!r}")
        source_path = _safe_source_path(source_dir, relative_path)
        observed_hash = _sha256(source_path)
        if observed_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {logical_name!r}: {observed_hash} != {expected_hash}")
        destination_path = destination_dir / source_path.name
        _atomic_copy(source_path, destination_path)
        compact_descriptors[logical_name] = {
            **descriptor,
            "path": destination_path.name,
            "sha256": observed_hash,
        }

    omitted = {name: descriptor for name, descriptor in source_descriptors.items() if name not in REPORT_ARTIFACTS}
    v0_sha256 = configuration.get("v0_parquet_sha256")
    compact_manifest = {
        **manifest,
        "cache": _portable_cache_identity(manifest.get("cache")),
        "v0_parquet": {
            "standard": "static-embedding-v0",
            "sha256": v0_sha256,
        }
        if isinstance(v0_sha256, str)
        else None,
        "artifacts": compact_descriptors,
        "study_state": state_descriptor,
        "publication_bundle": {
            "schema_version": "topocap-compact-publication-v1",
            "content": "portable manifest, study state, and compact report tables",
            "omitted_checkpoint_directories": ["models", "trials"],
            "omitted_artifacts": sorted(omitted),
        },
    }
    if omitted:
        compact_manifest["unpublished_artifacts"] = omitted
    destination_manifest = destination_dir / "manifest.json"
    _atomic_json(destination_manifest, compact_manifest)
    return destination_manifest


def main() -> int:
    args = _parse_args()
    destination = package_results(args.source_dir, args.destination_dir)
    print(f"published compact bundle: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
