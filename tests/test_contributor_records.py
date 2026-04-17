import pytest

from squadds.database.contributor_records import (
    add_sim_result_entry,
    build_contribution_payload,
    build_empty_contribution_state,
    merge_contributor_notes,
    validate_required_structure,
)


def test_build_empty_contribution_state_matches_legacy_defaults():
    state = build_empty_contribution_state()

    assert state == {
        "sim_results": {},
        "design": {"design_tool": "", "design_options": {}},
        "sim_options": {"setup": {}, "simulator": ""},
        "units": set(),
        "notes": {},
    }


def test_add_sim_result_entry_tracks_individual_units():
    sim_results, units = add_sim_result_entry({}, set(), "g_MHz", 55.0, "MHz")

    assert sim_results == {"g_MHz": 55.0, "g_MHz_unit": "MHz"}
    assert units == {"MHz"}


def test_build_contribution_payload_collapses_single_common_unit():
    payload = build_contribution_payload(
        {"design_tool": "", "design_options": {}},
        {"setup": {}, "simulator": ""},
        {"g_MHz": 55.0, "g_MHz_unit": "MHz", "kappa_kHz": 140.0, "kappa_kHz_unit": "MHz"},
        {"uploader": "shanto"},
        {},
        {"MHz"},
    )

    assert payload["sim_results"] == {"g_MHz": 55.0, "kappa_kHz": 140.0, "units": "MHz"}


def test_merge_contributor_notes_updates_existing_mapping():
    notes = merge_contributor_notes({"source": "paper"}, {"device": "chip-1"})

    assert notes == {"source": "paper", "device": "chip-1"}


def test_merge_contributor_notes_rejects_non_dict():
    with pytest.raises(ValueError, match="Notes must be provided as a dictionary."):
        merge_contributor_notes({}, "bad")


def test_validate_required_structure_checks_top_level_and_nested_keys():
    validate_required_structure(
        {"design": {"design_options": {}, "design_tool": ""}, "sim_results": {}, "notes": {}},
        {"design": {"design_options": "dict"}, "sim_results": "dict", "notes": "dict"},
    )


def test_validate_required_structure_raises_for_missing_nested_key():
    with pytest.raises(ValueError, match="Missing required sub-key 'design_options' in 'design'"):
        validate_required_structure(
            {"design": {"design_tool": ""}},
            {"design": {"design_options": "dict", "design_tool": "str"}},
        )
