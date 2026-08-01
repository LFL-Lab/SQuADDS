"""Compact TopoCap publication-bundle tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "package_topocap_report_results.py"


def _load_packager():
    specification = importlib.util.spec_from_file_location("package_topocap_report_results", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _study(tmp_path: Path, *, status: str = "EXPLORATORY_COMPLETE") -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    artifacts = {}
    for logical_name in _load_packager().REPORT_ARTIFACTS:
        path = source / f"{logical_name}.csv"
        path.write_text("value\n1\n")
        artifacts[logical_name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "rows": 1,
            "schema_version": "test-v1",
        }
    raw = source / "raw_trials.jsonl"
    raw.write_text('{"trial": 1}\n')
    artifacts["raw_trials"] = {"path": raw.name, "sha256": _sha256(raw), "rows": 1}
    state_digest = "a" * 64
    (source / "study-state.json").write_text(
        json.dumps(
            {
                "digest": state_digest,
                "configuration": {
                    "cache": {
                        "path": "/private/tmp/cache/graphs.jsonl",
                        "manifest_path": "/private/tmp/cache/manifest.json",
                        "graph_jsonl_sha256": "b" * 64,
                    },
                    "v0_parquet_sha256": "c" * 64,
                },
            }
        )
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test-v1",
                "study_status": status,
                "state_digest": state_digest,
                "cache": {
                    "path": "/private/tmp/cache/graphs.jsonl",
                    "manifest_path": "/private/tmp/cache/manifest.json",
                    "graph_jsonl_sha256": "b" * 64,
                },
                "v0_parquet": "/Users/example/.cache/static-embedding-v0.parquet",
                "artifacts": artifacts,
            }
        )
    )
    return source


def test_packager_verifies_and_omits_large_checkpoint_artifacts(tmp_path):
    packager = _load_packager()
    source = _study(tmp_path)
    destination = tmp_path / "publication"

    manifest_path = packager.package_results(source, destination)
    manifest = json.loads(manifest_path.read_text())

    assert set(manifest["artifacts"]) == set(packager.REPORT_ARTIFACTS)
    assert manifest["publication_bundle"]["omitted_artifacts"] == ["raw_trials"]
    assert "raw_trials" in manifest["unpublished_artifacts"]
    assert manifest["v0_parquet"] == {
        "sha256": "c" * 64,
        "standard": "static-embedding-v0",
    }
    assert "path" not in manifest["cache"]
    assert "manifest_path" not in manifest["cache"]
    state = json.loads((destination / "study-state.json").read_text())
    assert "path" not in state["configuration"]["cache"]
    assert "manifest_path" not in state["configuration"]["cache"]
    assert _sha256(destination / "study-state.json") == manifest["study_state"]["sha256"]
    assert not (destination / "raw_trials.jsonl").exists()
    for descriptor in manifest["artifacts"].values():
        path = destination / descriptor["path"]
        assert path.is_file()
        assert _sha256(path) == descriptor["sha256"]


def test_packager_rejects_quick_or_tampered_studies(tmp_path):
    packager = _load_packager()
    quick = _study(tmp_path / "quick", status="QUICK_SMOKE_NON_CLAIM_BEARING")
    with pytest.raises(ValueError, match="not publishable"):
        packager.package_results(quick, tmp_path / "quick-output")

    source = _study(tmp_path / "tampered")
    (source / "learning_curves.csv").write_text("value\nchanged\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        packager.package_results(source, tmp_path / "tampered-output")
