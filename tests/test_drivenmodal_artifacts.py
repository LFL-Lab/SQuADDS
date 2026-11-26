import json

from squadds.simulations.drivenmodal.artifacts import (
    initialize_run_manifest,
    load_run_manifest,
    mark_stage_complete,
    mark_sweep_point_failed,
    mark_sweep_point_success,
    register_sweep_points,
    select_sweep_points,
)


def test_initialize_run_manifest_creates_manifest_file(tmp_path):
    manifest = initialize_run_manifest(
        tmp_path,
        run_id="demo-run",
        request_payload={"system_kind": "ncap"},
    )

    manifest_path = tmp_path / "demo-run" / "manifest.json"

    assert manifest_path.exists()
    assert manifest["run_id"] == "demo-run"
    assert manifest["stages"]["prepared"]["status"] == "pending"
    assert json.loads(manifest_path.read_text())["request_payload"]["system_kind"] == "ncap"


def test_register_and_resume_sweep_points(tmp_path):
    initialize_run_manifest(
        tmp_path,
        run_id="demo-run",
        request_payload={"system_kind": "ncap"},
    )
    manifest = register_sweep_points(
        tmp_path / "demo-run" / "manifest.json",
        [
            {"point_id": "p1", "params": {"cap_gap": 10}},
            {"point_id": "p2", "params": {"cap_gap": 12}},
        ],
    )

    pending = select_sweep_points(manifest, statuses={"pending"})
    assert [point["point_id"] for point in pending] == ["p1", "p2"]

    manifest = mark_sweep_point_success(
        tmp_path / "demo-run" / "manifest.json",
        point_id="p1",
        artifact_uri="artifacts/drivenmodal/demo-run/p1.json",
    )
    manifest = mark_sweep_point_failed(
        tmp_path / "demo-run" / "manifest.json",
        point_id="p2",
        error_message="HFSS crashed",
    )

    pending_after = select_sweep_points(manifest, statuses={"pending"})
    failed_after = select_sweep_points(manifest, statuses={"failed"})

    assert pending_after == []
    assert [point["point_id"] for point in failed_after] == ["p2"]


def test_mark_stage_complete_updates_manifest_on_disk(tmp_path):
    initialize_run_manifest(
        tmp_path,
        run_id="demo-run",
        request_payload={"system_kind": "qubit_claw"},
    )

    updated = mark_stage_complete(tmp_path / "demo-run" / "manifest.json", "prepared")
    reloaded = load_run_manifest(tmp_path / "demo-run" / "manifest.json")

    assert updated["stages"]["prepared"]["status"] == "complete"
    assert reloaded["stages"]["prepared"]["status"] == "complete"


def test_initialize_run_manifest_resumes_existing_stages(tmp_path):
    initialize_run_manifest(tmp_path, run_id="r", request_payload={"a": 1})
    mark_stage_complete(tmp_path / "r", "sweep_completed")

    resumed = initialize_run_manifest(tmp_path, run_id="r", request_payload={"a": 2}, resume_existing=True)
    assert resumed["stages"]["sweep_completed"]["status"] == "complete"
    assert resumed["request_payload"] == {"a": 2}

    fresh = initialize_run_manifest(tmp_path, run_id="r", request_payload={"a": 2})
    assert fresh["stages"]["sweep_completed"]["status"] == "pending"


def test_run_drivenmodal_request_resumes_when_policy_allows(tmp_path):
    from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request
    from squadds.simulations.drivenmodal.models import (
        CapacitanceExtractionRequest,
        DrivenModalArtifactPolicy,
        DrivenModalLayerStackSpec,
        DrivenModalSetupSpec,
        DrivenModalSweepSpec,
    )

    def request(resume):
        return CapacitanceExtractionRequest(
            system_kind="qubit_claw",
            design_payload={"design_options": {}},
            layer_stack=DrivenModalLayerStackSpec(),
            setup=DrivenModalSetupSpec(),
            sweep=DrivenModalSweepSpec(start_ghz=1.0, stop_ghz=10.0, count=11),
            artifacts=DrivenModalArtifactPolicy(resume_existing=resume),
            metadata={"run_id": "resume-demo"},
        )

    run_drivenmodal_request(request(True), checkpoint_dir=tmp_path)
    mark_stage_complete(tmp_path / "resume-demo", "artifacts_exported")

    resumed = run_drivenmodal_request(request(True), checkpoint_dir=tmp_path)
    assert resumed["manifest"]["stages"]["artifacts_exported"]["status"] == "complete"

    restarted = run_drivenmodal_request(request(False), checkpoint_dir=tmp_path)
    assert restarted["manifest"]["stages"]["artifacts_exported"]["status"] == "pending"
