"""Pure helpers for contributor record state and validation."""

from __future__ import annotations


def build_empty_contribution_state():
    """Return the legacy empty contributor state blocks."""
    return {
        "sim_results": {},
        "design": {"design_tool": "", "design_options": {}},
        "sim_options": {"setup": {}, "simulator": ""},
        "units": set(),
        "notes": {},
    }


def add_sim_result_entry(sim_results: dict, units: set, result_name: str, result_value, unit: str):
    """Add a simulation result while preserving the legacy per-field unit behavior."""
    units.add(unit)
    sim_results[result_name] = result_value
    sim_results[f"{result_name}_unit"] = unit
    return sim_results, units


def build_contribution_payload(design, sim_options, sim_results, contributor, notes, units: set):
    """Build the serialized contribution payload using the legacy unit-collapsing behavior."""
    if len(units) == 1:
        common_unit = units.pop()
        sim_results["units"] = common_unit
        for result_name in list(sim_results.keys()):
            if "_unit" in result_name:
                del sim_results[result_name]
    return {
        "design": design,
        "sim_options": sim_options,
        "sim_results": sim_results,
        "contributor": contributor,
        "notes": notes,
    }


def merge_contributor_notes(existing_notes: dict, notes=None):
    """Merge notes with the legacy validation rules."""
    if notes is None:
        notes = {}
    if not isinstance(notes, dict):
        raise ValueError("Notes must be provided as a dictionary.")
    existing_notes.update(notes)
    return existing_notes


def validate_required_structure(actual_structure: dict, expected_structure: dict):
    """Validate top-level keys and one-level nested required keys."""
    for key, value in expected_structure.items():
        if key not in actual_structure:
            raise ValueError(f"Missing required key: {key}")
        if isinstance(value, dict):
            for sub_key in value:
                if sub_key not in actual_structure[key]:
                    raise ValueError(f"Missing required sub-key '{sub_key}' in '{key}'")
