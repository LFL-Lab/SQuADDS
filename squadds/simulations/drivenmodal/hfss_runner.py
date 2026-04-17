"""High-level driven-modal runner entrypoint."""

from __future__ import annotations

from pathlib import Path

from .artifacts import initialize_run_manifest, mark_stage_complete
from .layer_stack import resolve_layer_stack


def run_drivenmodal_request(request, *, checkpoint_dir: str | Path | None = None, export_artifacts: bool = True):
    """Initialize a checkpointed driven-modal run and return its prepared state."""
    request_payload = request.to_dict()
    resolved_layer_stack = resolve_layer_stack(request.layer_stack)
    root_dir = Path(checkpoint_dir or ".squadds_drivenmodal_runs")
    run_id = request.metadata.get("run_id", f"{request.__class__.__name__}-run")
    manifest = initialize_run_manifest(root_dir, run_id=run_id, request_payload=request_payload)
    manifest = mark_stage_complete(root_dir / run_id / "manifest.json", "prepared")

    return {
        "request": request_payload,
        "layer_stack": resolved_layer_stack,
        "manifest": manifest,
        "export_artifacts": export_artifacts,
    }
