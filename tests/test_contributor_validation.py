from squadds.database.contributor_validation import (
    find_common_key_type_matches,
    get_nested_value,
    summarize_content_differences,
)


def test_get_nested_value_returns_none_for_missing_paths():
    data = {"design": {"design_options": {"cross_length": "200um"}}}

    assert get_nested_value(data, "design.design_options.cross_length") == "200um"
    assert get_nested_value(data, "design.sim_options") is None


def test_find_common_key_type_matches_marks_mismatches_and_missing_paths():
    data = {"design": {"design_options": {"cross_length": "200um"}}, "notes": {}}
    ref = {"design": {"design_options": {"cross_length": 200.0}, "design_tool": "QM"}}

    matches = list(find_common_key_type_matches(data, ref))

    assert ("design.design_options.cross_length", False) in matches
    assert ("design.design_tool", None) in matches
    assert ("notes", None) in matches


def test_summarize_content_differences_groups_mismatched_and_missing_keys():
    data = {"design": {"design_options": {"cross_length": "200um"}}, "notes": {}}
    ref = {"design": {"design_options": {"cross_length": 200.0}, "design_tool": "QM"}}

    mismatched_keys, missing_keys = summarize_content_differences(data, ref)

    assert mismatched_keys == ["design.design_options.cross_length"]
    assert sorted(missing_keys) == ["design.design_tool", "notes"]
