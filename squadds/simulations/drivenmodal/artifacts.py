"""Checkpoint and progress-tracking helpers for driven-modal runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

STAGE_NAMES = [
    "prepared",
    "rendered",
    "setup_created",
    "sweep_completed",
    "artifacts_exported",
    "postprocessed",
    "serialized",
]


def _manifest_file(manifest_path_or_dir: str | Path) -> Path:
    manifest_path = Path(manifest_path_or_dir)
    if manifest_path.is_dir():
        return manifest_path / "manifest.json"
    return manifest_path


def _write_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def initialize_run_manifest(root_dir: str | Path, run_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    """Create a new run directory with an initial manifest."""
    run_dir = Path(root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "request_payload": request_payload,
        "stages": {stage_name: {"status": "pending"} for stage_name in STAGE_NAMES},
        "sweep_points": [],
    }
    return _write_manifest(manifest, manifest_path)


def load_run_manifest(manifest_path_or_dir: str | Path) -> dict[str, Any]:
    """Load a manifest from disk."""
    manifest_path = _manifest_file(manifest_path_or_dir)
    return json.loads(manifest_path.read_text())


def mark_stage_complete(manifest_path_or_dir: str | Path, stage_name: str) -> dict[str, Any]:
    """Mark a manifest stage complete and persist the change."""
    manifest_path = _manifest_file(manifest_path_or_dir)
    manifest = load_run_manifest(manifest_path)
    if stage_name not in manifest["stages"]:
        raise ValueError(f"Unknown stage_name: {stage_name}")
    manifest["stages"][stage_name]["status"] = "complete"
    return _write_manifest(manifest, manifest_path)


def register_sweep_points(manifest_path_or_dir: str | Path, sweep_points: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge new sweep points into a manifest without duplicating existing point ids."""
    manifest_path = _manifest_file(manifest_path_or_dir)
    manifest = load_run_manifest(manifest_path)
    existing_ids = {point["point_id"] for point in manifest["sweep_points"]}
    for point in sweep_points:
        point_id = point["point_id"]
        if point_id in existing_ids:
            continue
        manifest["sweep_points"].append(
            {
                "point_id": point_id,
                "params": deepcopy(point.get("params", {})),
                "status": "pending",
                "attempt_count": 0,
                "error_message": "",
                "artifact_uri": "",
            }
        )
        existing_ids.add(point_id)
    return _write_manifest(manifest, manifest_path)


def select_sweep_points(manifest: dict[str, Any], statuses: set[str]) -> list[dict[str, Any]]:
    """Select sweep points by status from an in-memory manifest."""
    return [point for point in manifest["sweep_points"] if point["status"] in statuses]


def _update_sweep_point(
    manifest_path_or_dir: str | Path,
    point_id: str,
    *,
    status: str,
    artifact_uri: str = "",
    error_message: str = "",
    increment_attempt: bool = False,
) -> dict[str, Any]:
    manifest_path = _manifest_file(manifest_path_or_dir)
    manifest = load_run_manifest(manifest_path)
    for point in manifest["sweep_points"]:
        if point["point_id"] != point_id:
            continue
        point["status"] = status
        point["artifact_uri"] = artifact_uri
        point["error_message"] = error_message
        if increment_attempt:
            point["attempt_count"] += 1
        return _write_manifest(manifest, manifest_path)

    raise ValueError(f"Unknown point_id: {point_id}")


def mark_sweep_point_running(manifest_path_or_dir: str | Path, point_id: str) -> dict[str, Any]:
    """Mark a sweep point as running and increment attempts."""
    return _update_sweep_point(
        manifest_path_or_dir,
        point_id,
        status="running",
        increment_attempt=True,
    )


def mark_sweep_point_success(
    manifest_path_or_dir: str | Path,
    point_id: str,
    artifact_uri: str,
) -> dict[str, Any]:
    """Mark a sweep point successful and persist its artifact URI."""
    return _update_sweep_point(
        manifest_path_or_dir,
        point_id,
        status="success",
        artifact_uri=artifact_uri,
    )


def mark_sweep_point_failed(
    manifest_path_or_dir: str | Path,
    point_id: str,
    error_message: str,
) -> dict[str, Any]:
    """Mark a sweep point failed and persist the latest error message."""
    return _update_sweep_point(
        manifest_path_or_dir,
        point_id,
        status="failed",
        error_message=error_message,
    )
