import glob
import json
import os


def append_entries_to_dataset_file(dataset_file, entries):
    """
    Append one or more contribution entries to a dataset JSON file.
    """
    with open(dataset_file, "r+") as file:
        data = json.load(file)
        data.extend(entries)
        file.seek(0)
        json.dump(data, file, indent=4)


def load_contribution_from_json_file(file_path):
    """
    Load a single contribution entry from disk.
    """
    with open(file_path) as file:
        return json.load(file)


def load_sweep_entries_from_json_prefix(json_prefix, contributor_info):
    """
    Load all sweep entries under a JSON filename prefix.
    """
    entries = []
    for file_path in sorted(glob.glob(os.path.abspath(json_prefix + "*.json"))):
        with open(file_path) as file:
            data = json.load(file)
        entry = {
            "design": data["design"],
            "sim_options": data["sim_options"],
            "sim_results": data["sim_results"],
            "contributor": contributor_info,
            "notes": data.get("notes", {}),
        }
        entries.append(entry)
    return entries


def validate_sweep_entries(entries, validate_structure_fn, validate_types_fn, validate_content_fn, print_fn=print):
    """
    Run the legacy validation loop over a sweep payload.
    """
    total_entries = len(entries)
    for index, entry in enumerate(entries, start=1):
        print_fn(f"Validating entry {index} of {total_entries}...")
        validate_structure_fn(entry)
        validate_types_fn(entry)
        validate_content_fn(entry)
        print_fn(f"Entry {index} of {total_entries} validated successfully.")
        print_fn("--------------------------------------------------")
