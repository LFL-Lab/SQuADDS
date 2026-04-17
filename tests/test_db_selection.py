import pytest

from squadds.core.db_selection import (
    build_component_selection,
    is_supported_coupler,
    resolve_resonator_coupler,
    resolve_system_selection,
)


def test_resolve_system_selection_accepts_single_component():
    system, component, unsupported = resolve_system_selection("qubit", ["qubit", "cavity_claw"])

    assert system == "qubit"
    assert component == "qubit"
    assert unsupported is None


def test_resolve_system_selection_accepts_component_lists():
    system, component, unsupported = resolve_system_selection(["qubit", "cavity_claw"], ["qubit", "cavity_claw"])

    assert system == ["qubit", "cavity_claw"]
    assert component is None
    assert unsupported is None


def test_resolve_system_selection_reports_first_unsupported_component():
    system, component, unsupported = resolve_system_selection(["qubit", "resonator"], ["qubit", "cavity_claw"])

    assert system is None
    assert component is None
    assert unsupported == "resonator"


def test_build_component_selection_returns_component_name_and_data_type():
    component_name, data_type = build_component_selection(
        ["qubit", "cavity_claw"],
        "qubit",
        "TransmonCross",
        "cap_matrix",
        "warning",
    )

    assert component_name == "TransmonCross"
    assert data_type == "cap_matrix"


def test_build_component_selection_raises_for_mismatched_system():
    with pytest.raises(UserWarning, match="warning"):
        build_component_selection("cavity_claw", "qubit", "TransmonCross", "cap_matrix", "warning")


def test_resolve_resonator_coupler_maps_legacy_aliases():
    assert resolve_resonator_coupler("quarter") == "CLT"
    assert resolve_resonator_coupler("half") == "NCap"


def test_resolve_resonator_coupler_rejects_unknown_values():
    with pytest.raises(ValueError, match="Invalid resonator type"):
        resolve_resonator_coupler("lumped")


def test_is_supported_coupler_allows_clt_and_known_component_names():
    assert is_supported_coupler("CLT", ["NCap"])
    assert is_supported_coupler("NCap", ["NCap"])
    assert not is_supported_coupler("Unknown", ["NCap"])
