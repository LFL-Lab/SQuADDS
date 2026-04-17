import json

from squadds.database.contributor_file_ops import (
    append_entries_to_dataset_file,
    load_contribution_from_json_file,
    load_sweep_entries_from_json_prefix,
    validate_sweep_entries,
)


def test_append_entries_to_dataset_file_extends_existing_json_array(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(json.dumps([{"value": 1}]))

    append_entries_to_dataset_file(str(dataset_file), [{"value": 2}, {"value": 3}])

    assert json.loads(dataset_file.read_text()) == [{"value": 1}, {"value": 2}, {"value": 3}]


def test_load_contribution_from_json_file_reads_single_payload(tmp_path):
    payload_file = tmp_path / "entry.json"
    payload_file.write_text(json.dumps({"design": {}, "sim_options": {}, "sim_results": {}}))

    assert load_contribution_from_json_file(str(payload_file)) == {
        "design": {},
        "sim_options": {},
        "sim_results": {},
    }


def test_load_sweep_entries_from_json_prefix_builds_contributor_enriched_entries(tmp_path):
    first = tmp_path / "sweep_1.json"
    second = tmp_path / "sweep_2.json"
    first.write_text(json.dumps({"design": {"a": 1}, "sim_options": {"b": 2}, "sim_results": {"c": 3}}))
    second.write_text(json.dumps({"design": {"a": 4}, "sim_options": {"b": 5}, "sim_results": {"c": 6}, "notes": {"d": 7}}))

    entries = load_sweep_entries_from_json_prefix(str(tmp_path / "sweep_"), {"uploader": "shanto"})

    assert entries == [
        {
            "design": {"a": 1},
            "sim_options": {"b": 2},
            "sim_results": {"c": 3},
            "contributor": {"uploader": "shanto"},
            "notes": {},
        },
        {
            "design": {"a": 4},
            "sim_options": {"b": 5},
            "sim_results": {"c": 6},
            "contributor": {"uploader": "shanto"},
            "notes": {"d": 7},
        },
    ]


def test_validate_sweep_entries_runs_all_callbacks_in_order():
    seen = []

    def record(name):
        return lambda entry: seen.append((name, entry["id"]))

    printed = []
    validate_sweep_entries(
        [{"id": 1}, {"id": 2}],
        validate_structure_fn=record("structure"),
        validate_types_fn=record("types"),
        validate_content_fn=record("content"),
        print_fn=printed.append,
    )

    assert seen == [
        ("structure", 1),
        ("types", 1),
        ("content", 1),
        ("structure", 2),
        ("types", 2),
        ("content", 2),
    ]
    assert printed[0] == "Validating entry 1 of 2..."
    assert printed[-1] == "--------------------------------------------------"
