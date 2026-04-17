import pytest

from squadds.simulations.drivenmodal.models import (
    CapacitanceExtractionRequest,
    CoupledSystemDrivenModalRequest,
    DrivenModalArtifactPolicy,
    DrivenModalLayerStackSpec,
    DrivenModalSetupSpec,
    DrivenModalSweepSpec,
)


def test_capacitance_request_rejects_unknown_system_kind():
    with pytest.raises(ValueError, match="system_kind"):
        CapacitanceExtractionRequest(
            system_kind="bad_kind",
            design_payload={"design_options": {}},
            layer_stack=DrivenModalLayerStackSpec(),
            setup=DrivenModalSetupSpec(),
            sweep=DrivenModalSweepSpec(start_ghz=1.0, stop_ghz=10.0, count=101),
            artifacts=DrivenModalArtifactPolicy(),
        )


def test_coupled_request_rejects_unknown_resonator_type():
    with pytest.raises(ValueError, match="resonator_type"):
        CoupledSystemDrivenModalRequest(
            resonator_type="bad_type",
            design_payload={"design_options": {}},
            layer_stack=DrivenModalLayerStackSpec(),
            setup=DrivenModalSetupSpec(),
            sweep=DrivenModalSweepSpec(start_ghz=4.0, stop_ghz=9.0, count=801),
            artifacts=DrivenModalArtifactPolicy(),
        )


def test_sweep_spec_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="stop_ghz"):
        DrivenModalSweepSpec(start_ghz=8.0, stop_ghz=8.0, count=101)

    with pytest.raises(ValueError, match="count"):
        DrivenModalSweepSpec(start_ghz=4.0, stop_ghz=8.0, count=1)


def test_request_to_dict_includes_nested_specs():
    request = CapacitanceExtractionRequest(
        system_kind="ncap",
        design_payload={"design_options": {"finger_count": 6}},
        layer_stack=DrivenModalLayerStackSpec(),
        setup=DrivenModalSetupSpec(),
        sweep=DrivenModalSweepSpec(start_ghz=1.0, stop_ghz=10.0, count=101),
        artifacts=DrivenModalArtifactPolicy(export_touchstone=True),
    )

    payload = request.to_dict()

    assert payload["system_kind"] == "ncap"
    assert payload["layer_stack"]["preset"] == "squadds_hfss_v1"
    assert payload["artifacts"]["export_touchstone"] is True
