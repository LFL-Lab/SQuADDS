"""Pure helpers for contributor content comparison and reporting."""

from __future__ import annotations


def get_nested_value(dictionary, keys):
    """Walk a dotted path through a nested dictionary and return `None` on missing keys."""
    for key in keys.split("."):
        if dictionary is not None and key in dictionary:
            dictionary = dictionary[key]
        else:
            return None
    return dictionary


def find_common_key_type_matches(dict1, dict2, path=""):
    """Yield dotted-key comparisons for matching and missing nested paths."""
    common_keys = set(dict1.keys()) & set(dict2.keys())
    diff_keys = (set(dict1.keys()) - set(dict2.keys())) | (set(dict2.keys()) - set(dict1.keys()))
    for key in common_keys:
        new_path = f"{path}.{key}" if path else key
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            yield from find_common_key_type_matches(dict1[key], dict2[key], new_path)
        else:
            if type(dict1[key]) != type(dict2[key]):
                yield new_path, False
            else:
                yield new_path, True
    for key in diff_keys:
        new_path = f"{path}.{key}" if path else key
        yield new_path, None


def summarize_content_differences(data: dict, ref: dict):
    """Return categorized dotted-key differences between contributor data and a reference entry."""
    result = list(find_common_key_type_matches(data, ref))
    mismatched_keys = [key for key, match in result if match is False]
    missing_keys = [key for key, match in result if match is None]
    return mismatched_keys, missing_keys
